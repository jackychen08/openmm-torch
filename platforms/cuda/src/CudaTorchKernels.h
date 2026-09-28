#ifndef CUDA_TORCH_KERNELS_H_
#define CUDA_TORCH_KERNELS_H_

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

#include "TorchKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <torch/version.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <optional>
#include <set>

namespace TorchPlugin {

/**
 * This kernel is invoked by TorchForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcTorchForceKernel : public CalcTorchForceKernel {
public:
    CudaCalcTorchForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu);
    ~CudaCalcTorchForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system         the System this kernel will be applied to
     * @param force          the TorchForce this kernel will be used for
     * @param module         the PyTorch module to use for computing forces and energy
     */
    void initialize(const OpenMM::System& system, const TorchForce& force, torch::jit::script::Module& module);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

private:
    bool hasInitializedKernel;
    OpenMM::CudaContext& cu;
    torch::jit::script::Module module;
    torch::Tensor posTensor, velTensor, boxTensor;
    torch::Tensor energyTensor, forceTensor;
    std::map<std::string, torch::Tensor> globalTensors;
    std::vector<std::string> globalNames;
    std::set<std::string> paramDerivs;
    bool usePeriodic, outputsForces, useVelocities;
    std::vector<torch::Tensor> externalTensors;
    CUfunction copyInputsKernel, copyVelocitiesKernel, addForcesKernel;
    CUcontext primaryContext;
    // useOpenMMStream: PyTorch runs on its own stream, ordered against OpenMM's stream by events, with no host waits.
    std::optional<c10::cuda::CUDAStream> torchStream;
    CUevent inputsReady = nullptr, modelDone = nullptr;
    torch::Tensor convertedForces;       // stream mode: the forces in OpenMM's precision, alive until the next call
    std::map<bool, at::cuda::CUDAGraph> graphs;
    // Each graph's own output tensors (energy, force), keyed like graphs. A capture assigns energyTensor/forceTensor
    // to that graph's outputs, so a replay must read back the outputs of the graph it replayed, not of the last one
    // captured: an energy-only request once made every later force evaluation add the energy graph's stale forces.
    std::map<bool, std::pair<torch::Tensor, torch::Tensor>> graphOutputs;
    void prepareTorchInputs(OpenMM::ContextImpl& context, std::vector<torch::jit::IValue>& inputs, std::map<std::string, torch::Tensor>& derivInputs);
    bool useGraphs, useOpenMMStream;
    void addForces(torch::Tensor& forceTensor);
    int warmupSteps;
};

} // namespace TorchPlugin

#endif /*CUDA_TORCH_KERNELS_H_*/
