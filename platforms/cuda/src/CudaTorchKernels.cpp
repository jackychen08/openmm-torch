/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2024 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors: Raimondas Galvelis, Raul P. Pelaez                           *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaTorchKernels.h"
#include <optional>
#include "CudaTorchKernelSources.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

// macro for checking the result of synchronization operation on CUDA
// copied from `openmm/platforms/cuda/src/CudaParallelKernels.cpp`
#define CHECK_RESULT(result, prefix)                                             \
    if (result != CUDA_SUCCESS) {                                                \
        std::stringstream m;                                                     \
        m << prefix << ": " << cu.getErrorString(result) << " (" << result << ")"\
          << " at " << __FILE__ << ":" << __LINE__;                              \
        throw OpenMMException(m.str());                                          \
    }

CudaCalcTorchForceKernel::CudaCalcTorchForceKernel(string name, const Platform& platform, CudaContext& cu) : CalcTorchForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    // Explicitly activate the primary context
    CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext, cu.getDevice()), "Failed to retain the primary context");
}

CudaCalcTorchForceKernel::~CudaCalcTorchForceKernel() {
    if (inputsReady != nullptr)
        cuEventDestroy(inputsReady);
    if (modelDone != nullptr)
        cuEventDestroy(modelDone);
    cuDevicePrimaryCtxRelease(cu.getDevice());
}

void CudaCalcTorchForceKernel::initialize(const System& system, const TorchForce& force, torch::jit::script::Module& module) {
    this->module = module;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    outputsForces = force.getOutputsForces();
    useVelocities = force.usesVelocities();
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalNames.push_back(force.getGlobalParameterName(i));
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        paramDerivs.insert(force.getEnergyParameterDerivativeName(i));
        cu.addEnergyParameterDerivative(force.getEnergyParameterDerivativeName(i));
    }
    int numParticles = system.getNumParticles();

    // Push the PyTorch context
    // NOTE: Pytorch is always using the primary context.
    //       It makes the primary context current, if it is not a case.
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

    // Initialize CUDA objects for PyTorch
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex()); // This implicitly initialize PyTorch
    this->module.to(device);
    this->module.eval();
    this->module = torch::jit::freeze(this->module);
    torch::TensorOptions options = torch::TensorOptions().device(device).dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
    posTensor = torch::empty({numParticles, 3}, options.requires_grad(!outputsForces));
    // Velocities are passed in double precision on every platform: in mixed precision OpenMM stores them as double.
    velTensor = torch::empty({numParticles, 3}, options.dtype(torch::kFloat64));
    boxTensor = torch::empty({3, 3}, options);
    for (int i = 0; i < force.getNumExternalTensors(); i++) {
        // Only device memory of this context's device is accepted; anything else would be read by the GPU as garbage.
        CUdeviceptr pointer = (CUdeviceptr) force.getExternalTensorPointer(i);
        unsigned int memoryType = 0;
        int ordinal = -1;
        if (cuPointerGetAttribute(&memoryType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer) != CUDA_SUCCESS ||
                memoryType != CU_MEMORYTYPE_DEVICE ||
                cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pointer) != CUDA_SUCCESS ||
                ordinal != cu.getDeviceIndex())
            throw OpenMMException("TorchForce: external tensor '" + force.getExternalTensorName(i) +
                                  "' is not device memory on the context's device");
        externalTensors.push_back(torch::from_blob(reinterpret_cast<void*>(pointer),
                                                   {(int64_t) force.getExternalTensorSize(i)},
                                                   torch::TensorOptions().device(device).dtype(torch::kFloat32)));
    }
    energyTensor = torch::empty({0}, options);
    forceTensor = torch::empty({0}, options);
    for (const string& name : globalNames)
        globalTensors[name] = torch::tensor({0}, options);
    // Pop the PyToch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that PyTorch haven't messed up the context stack

    // Initialize CUDA objects for OpenMM-Torch
    ContextSelector selector(cu); // Switch to the OpenMM context
    map<string, string> defines;
    CUmodule program = cu.createModule(CudaTorchKernelSources::torchForce, defines);
    copyInputsKernel = cu.getKernel(program, "copyInputs");
    copyVelocitiesKernel = cu.getKernel(program, "copyVelocities");
    addForcesKernel = cu.getKernel(program, "addForces");
    auto properties = force.getProperties();
    const std::string useCUDAGraphsString = properties["useCUDAGraphs"];
    if (useCUDAGraphsString == "true")
        useGraphs = true;
    else if (useCUDAGraphsString == "false" || useCUDAGraphsString == "")
        useGraphs = false;
    else
        throw OpenMMException("TorchForce: invalid value of \"useCUDAGraphs\"");
    if (useGraphs) {
        this->warmupSteps = std::stoi(properties["CUDAGraphWarmupSteps"]);
        if (this->warmupSteps <= 0) {
            throw OpenMMException("TorchForce: \"CUDAGraphWarmupSteps\" must be a positive integer");
        }
    }
    // useOpenMMStream: no host waits inside a call. OpenMM runs in its own CUDA context and PyTorch in the device's
    // primary context, so their streams cannot be shared; instead an event recorded on OpenMM's stream after the
    // inputs are copied gates PyTorch's stream, and an event recorded on PyTorch's stream after the model gates
    // OpenMM's stream before the forces are added (a stream may wait on an event of another context on the device).
    // The energy is read only when requested, since reading it waits for the model.
    const std::string useOpenMMStreamString = properties["useOpenMMStream"];
    if (useOpenMMStreamString == "true")
        useOpenMMStream = true;
    else if (useOpenMMStreamString == "false" || useOpenMMStreamString == "")
        useOpenMMStream = false;
    else
        throw OpenMMException("TorchForce: invalid value of \"useOpenMMStream\"");
    if (useOpenMMStream) {
        CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");
        torchStream = c10::cuda::getStreamFromPool(false, cu.getDeviceIndex());
        CHECK_RESULT(cuEventCreate(&modelDone, CU_EVENT_DISABLE_TIMING), "Failed to create a CUDA event");
        CUcontext ctx;
        CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
        ContextSelector selector(cu);
        CHECK_RESULT(cuEventCreate(&inputsReady, CU_EVENT_DISABLE_TIMING), "Failed to create a CUDA event");
    }
}

/**
 * Get a pointer to the data in a PyTorch tensor.
 * The tensor is converted to the correct data type if necessary.
 */
static void* getTensorPointer(OpenMM::CudaContext& cu, torch::Tensor& tensor) {
    void* data;
    if (cu.getUseDoublePrecision()) {
        data = tensor.to(torch::kFloat64).data_ptr<double>();
    } else {
        data = tensor.to(torch::kFloat32).data_ptr<float>();
    }
    return data;
}

/**
 * Prepare the inputs for the PyTorch model, copying positions from the OpenMM context.
 */
void CudaCalcTorchForceKernel::prepareTorchInputs(ContextImpl& context, vector<torch::jit::IValue>& inputs, map<string, torch::Tensor>& globalTensors) {
    int numParticles = cu.getNumAtoms();
    // Get pointers to the atomic positions and simulation box
    void* posData = getTensorPointer(cu, posTensor);
    void* boxData = getTensorPointer(cu, boxTensor);
    // Copy the atomic positions and simulation box to PyTorch tensors
    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        void* inputArgs[] = {&posData,
                             &boxData,
                             &cu.getPosq().getDevicePointer(),
                             &cu.getAtomIndexArray().getDevicePointer(),
                             &numParticles,
                             cu.getPeriodicBoxVecXPointer(),
                             cu.getPeriodicBoxVecYPointer(),
                             cu.getPeriodicBoxVecZPointer()};
        cu.executeKernel(copyInputsKernel, inputArgs, numParticles);
        if (useVelocities) {
            void* velData = velTensor.data_ptr<double>();
            void* velArgs[] = {&velData, &cu.getVelm().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles};
            cu.executeKernel(copyVelocitiesKernel, velArgs, numParticles);
        }
        if (useOpenMMStream) {
            CHECK_RESULT(cuEventRecord(inputsReady, cu.getCurrentStream()), "Failed to record a CUDA event");
        } else {
            CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
        }
    }
    if (useOpenMMStream) {
        CHECK_RESULT(cuStreamWaitEvent(torchStream->stream(), inputsReady, 0), "Failed to wait for a CUDA event");
    }
    // Prepare the input of the PyTorch model
    inputs = {posTensor};
    if (useVelocities)
        inputs.push_back(velTensor);
    if (usePeriodic)
        inputs.push_back(boxTensor);
    for (torch::Tensor& tensor : externalTensors)
        inputs.push_back(tensor);
    for (const string& name : globalNames) {
        // PyTorch requires us to set requires_grad to false before initializing a tensor.
        globalTensors[name].set_requires_grad(false);
        globalTensors[name][0] = context.getParameter(name);
        if (paramDerivs.find(name) != paramDerivs.end())
            globalTensors[name].set_requires_grad(true);
        inputs.push_back(globalTensors[name]);
    }
}

/**
 * Add the computed forces to the total atomic forces.
 */
void CudaCalcTorchForceKernel::addForces(torch::Tensor& forceTensor) {
    int numParticles = cu.getNumAtoms();
    // Get a pointer to the computed forces
    void* forceData = getTensorPointer(cu, forceTensor);
    if (!useOpenMMStream) {
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the OpenMM context
    }
    // Add the computed forces to the total atomic forces
    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        if (useOpenMMStream) {
            CHECK_RESULT(cuStreamWaitEvent(cu.getCurrentStream(), modelDone, 0), "Failed to wait for a CUDA event");
        }
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        int forceSign = (outputsForces ? 1 : -1);
        void* forceArgs[] = {&forceData, &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms, &forceSign};
        cu.executeKernel(addForcesKernel, forceArgs, numParticles);
        if (!useOpenMMStream) {
            CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
        }
    }
}

/**
 * This function launches the workload in a way compatible with CUDA
 * graphs as far as OpenMM-Torch goes.  Capturing this function when
 * the model is not itself graph compatible (due to, for instance,
 * implicit synchronizations) will result in a CUDA error.
 */
static void executeGraph(bool outputsForces, bool includeForces, torch::jit::script::Module& module, vector<torch::jit::IValue>& inputs, torch::Tensor& posTensor, torch::Tensor& energyTensor,
                         torch::Tensor& forceTensor, map<string, torch::Tensor>& globalTensors, set<string> paramDerivs) {
    vector<torch::Tensor> gradInputs;
    if (!outputsForces && includeForces)
        gradInputs.push_back(posTensor);
    for (auto& name : paramDerivs)
        gradInputs.push_back(globalTensors[name]);
    auto none = torch::Tensor();
    if (outputsForces) {
        auto outputs = module.forward(inputs).toTuple();
        energyTensor = outputs->elements()[0].toTensor();
        forceTensor = outputs->elements()[1].toTensor();
        if (gradInputs.size() > 0)
            energyTensor.backward(none, false, false, gradInputs);
    } else {
        energyTensor = module.forward(inputs).toTensor();
        // Compute force by backpropagating the PyTorch model
        // CUDA graph capture sometimes fails if backwards is not explicitly requested w.r.t positions
        // See https://github.com/openmm/openmm-torch/pull/120/
        if (gradInputs.size() > 0)
            energyTensor.backward(none, false, false, gradInputs);
        if (includeForces) {
            // This is minus the forces, we change the sign later on
            forceTensor = posTensor.grad().clone();
            // Zero the gradient to avoid accumulating it
            posTensor.grad().zero_();
        }
    }
}

double CudaCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // Push to the PyTorch context
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");
    std::optional<c10::cuda::CUDAStreamGuard> torchStreamGuard;
    if (useOpenMMStream)
        torchStreamGuard.emplace(*torchStream);
    vector<torch::jit::IValue> inputs;
    prepareTorchInputs(context, inputs, globalTensors);
    if (!useGraphs) {
        executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor, globalTensors, paramDerivs);
    } else {
        // Record graph if not already done
        bool is_graph_captured = false;
        if (graphs.find(includeForces) == graphs.end()) {
            // Warm-up and capture run on a side stream: in stream mode the inputs queued on OpenMM's stream must be
            // complete first, and the side stream's work before PyTorch's stream continues (once per graph).
            if (useOpenMMStream) {
                CHECK_RESULT(cuEventSynchronize(inputsReady), "Failed to wait for a CUDA event");
            }
	    //CUDA graph capture must occur in a non-default stream
            const auto stream = c10::cuda::getStreamFromPool(false, cu.getDeviceIndex());
	        const c10::cuda::CUDAStreamGuard guard(stream);
            // Warmup the graph workload before capturing.  This first
            // run  before  capture sets  up  allocations  so that  no
            // allocations are  needed after.  Pytorch's  allocator is
            // stream  capture-aware and,  after warmup,  will provide
            // record static pointers and shapes during capture.
            try {
                for (int i = 0; i < this->warmupSteps; i++)
                    executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor, globalTensors, paramDerivs);
            }
            catch (std::exception& e) {
                throw OpenMMException(string("TorchForce Failed to warmup the model before graph construction. Torch reported the following error:\n") + e.what());
            }
            graphs[includeForces].capture_begin();
            try {
                executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor, globalTensors, paramDerivs);
                is_graph_captured = true;
                graphs[includeForces].capture_end();
            }
            catch (std::exception& e) {
                if (!is_graph_captured) {
                    graphs[includeForces].capture_end();
                }
                throw OpenMMException(string("TorchForce Failed to capture the model into a CUDA graph. Torch reported the following error:\n") + e.what());
            }
            for (const string& name : paramDerivs)
                globalTensors[name].grad().zero_();
            if (useOpenMMStream) {
                CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context");
            }
        }
        if (useOpenMMStream)
            graphs[includeForces].replay();                 // on PyTorch's stream (the guard), gated by inputsReady
        else {
            // Use the same stream as the OpenMM context, even if it is the default stream
            const auto openmmStream = cu.getCurrentStream();
            const auto stream = c10::cuda::getStreamFromExternal(openmmStream, cu.getDeviceIndex());
            const c10::cuda::CUDAStreamGuard guard(stream);
            graphs[includeForces].replay();
        }
    }
    if (useOpenMMStream) {
        // Convert on PyTorch's stream before the event, into a member that outlives the kernel reading it on
        // OpenMM's stream. Not into forceTensor: with CUDA graphs that is the graph's output buffer, and replacing it
        // made every later call add the first call's forces (caught by test_velocities_reach_the_model_in_system_order
        // with a float64 model output). OpenMM's stream waits for the model even without forces, so its next input
        // copy cannot overwrite what the model reads.
        if (includeForces)
            convertedForces = forceTensor.to(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
        CHECK_RESULT(cuEventRecord(modelDone, torchStream->stream()), "Failed to record a CUDA event");
        if (!includeForces) {
            ContextSelector selector(cu);
            CHECK_RESULT(cuStreamWaitEvent(cu.getCurrentStream(), modelDone, 0), "Failed to wait for a CUDA event");
        }
    }
    if (includeForces) {
        addForces(useOpenMMStream ? convertedForces : forceTensor);
    }
    map<string, double>& energyParamDerivs = cu.getEnergyParamDerivWorkspace();
    for (const string& name : paramDerivs) {
        energyParamDerivs[name] += globalTensors[name].grad().item<double>();
        globalTensors[name].grad().zero_();
    }
    // Get energy. In stream mode only when it was requested: reading it waits for the whole stream.
    double energy = 0.0;
    if (!useOpenMMStream || includeEnergy)
        energy = energyTensor.item<double>(); // This implicitly synchronizes the PyTorch context
    torchStreamGuard.reset();
    // Pop to the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that the correct context was popped
    return energy;
}
