"""
Example: Genetic Algorithm for Optical Topology Search using LightwaveExplorer

This script demonstrates how to use the SimulationRunner to not only optimize 
standard scalar parameters (like pulse energy), but also to search for optimal
optical topologies where the parameters are embedded in the `sequence` string.

Scenario: Finding an optimal F-2F interferometer SHG stage.
Goal: Maximize the intensity of the SHG signal at a specific frequency range
      by optimizing both the crystal thickness and the phase matching angle 
      inside a custom generated `sequence` string.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import LightwaveExplorer as lwe

# =============================================================================
# 1. Define the Genetic Representation (Topology)
# =============================================================================

class TopologyParams:
    """Represents a set of parameters that define an optical topology."""
    def __init__(self, shg_theta, shg_thickness, pulse_energy):
        self.shg_theta = shg_theta            # BBO angle in degrees
        self.shg_thickness = shg_thickness    # BBO thickness in meters
        self.pulse_energy = pulse_energy      # Input pulse energy in Joules
        self.loss = float("inf")              # To be evaluated
        
    def generate_sequence(self):
        """
        CRITICAL PART: This is where we embed our parameters into the LWE sequence string!
        By dynamically generating this string, the GA can optimize these embedded values.
        """
        # Example sequence:
        # 1. crystal(4, theta, 0): Switch to BBO crystal at angle theta
        # 2. propagate(thickness): Propagate through the crystal
        # 3. defaultMaterial():    Switch back to vacuum/default
        # 4. save(1):              Save the result
        
        return (
            f"crystal(4, {self.shg_theta:.4f}, 0) "
            f"propagate({self.shg_thickness:.6e}) "
            "defaultMaterial() "
            "save(1)"
        )

    def mutate(self, sigma=0.1):
        """Create a randomly mutated child based on a Gaussian distribution."""
        return TopologyParams(
            shg_theta = self.shg_theta * (1 + np.random.normal(0, sigma)),
            shg_thickness = self.shg_thickness * (1 + np.random.normal(0, sigma)),
            pulse_energy = self.pulse_energy * (1 + np.random.normal(0, sigma))
        )

# =============================================================================
# 2. Define the Evaluation Function
# =============================================================================

def evaluate_topology(topo: TopologyParams, cli_path: str, work_dir: str):
    """
    Runs a single LWE simulation for a given topology and calculates the loss.
    This function is designed to run completely independently in a parallel process.
    """
    # Create isolated working directory to avoid parallel conflicts
    os.makedirs(work_dir, exist_ok=True)
    
    # Initialize runner targeting this specific isolated directory
    runner = lwe.SimulationRunner(cli_path=cli_path, work_dir=work_dir)
    
    try:
        # 1. Base settings (use a coarse grid for fast iteration in GA)
        runner.set_params(
            time_span=100e-15,
            dt=1e-15,            # Coarse dt
            grid_width=50e-6, 
            dx=5e-6,             # Coarse dx
            symmetry_type=1      # Cylindrical symmetry is faster
        )
        
        # 2. Inject our GA parameters
        #    Notice how we pass the generated sequence string!
        runner.set_params(
            sequence=topo.generate_sequence(),
            pulse_energy1=topo.pulse_energy
        )
        
        # 3. Run the simulation
        runner.run(verbose=False, timeout=120)
        
        # 4. Calculate Loss
        # Example objective: Maximize the peak intensity of the output spectrum
        # (Since we minimize loss, we use the negative peak intensity)
        spectrum = runner.result.spectrumTotal
        loss = -np.max(spectrum)
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        loss = float("inf")
    finally:
        # Cleanup isolated files
        runner.cleanup()
        shutil.rmtree(work_dir, ignore_errors=True)
        
    return loss

# =============================================================================
# 3. Run the Genetic Algorithm (Steady-State parallelized)
# =============================================================================

if __name__ == '__main__':
    # Configuration
    CLI_PATH = "./build_cli/LightwaveExplorer"  # Path to the compiled Linux CLI
    POP_SIZE = 8
    MAX_WORKERS = 4      # Number of parallel simulations (e.g. CPU core count)
    N_GENERATIONS = 10
    
    print(f"Starting Genetic Algorithm with {POP_SIZE} individuals for {N_GENERATIONS} generations...")
    
    # 3a. Initialize random population
    population = []
    for _ in range(POP_SIZE):
        child = TopologyParams(
            shg_theta=np.random.uniform(20.0, 30.0),       # Initial random guess
            shg_thickness=np.random.uniform(50e-6, 300e-6),
            pulse_energy=np.random.uniform(0.5e-6, 2e-6)
        )
        population.append(child)

    # 3b. Evolution Loop
    for gen in range(N_GENERATIONS):
        print(f"\n--- Generation {gen} ---")
        
        # We need to evaluate every topology that hasn't been evaluated yet
        unevaluated = [topo for topo in population if topo.loss == float("inf")]
        
        # Run parallel evaluations
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for i, topo in enumerate(unevaluated):
                # Unique temp dir for each parallel run
                work_dir = f"/tmp/lwe_ga/gen{gen}_{i}"
                future = pool.submit(evaluate_topology, topo, CLI_PATH, work_dir)
                futures[future] = topo
            
            # Wait for results and assign
            for future in as_completed(futures):
                topo = futures[future]
                topo.loss = future.result()
        
        # 3c. Sort population by fitness (lowest loss is best)
        population.sort(key=lambda x: x.loss)
        
        best = population[0]
        worst = population[-1]
        print(f"Best Loss: {best.loss:.4e} | Worst Loss: {worst.loss:.4e}")
        print(f"Best Params -> Theta: {best.shg_theta:.2f}, Thickness: {best.shg_thickness*1e6:.1f}um, Energy: {best.pulse_energy*1e6:.2f}uJ")
        print(f"Best Sequence: '{best.generate_sequence()}'")
        
        # 3d. Selection and Mutation (Steady-State replacement)
        # Select a parent (preferring those with lower loss using a Gaussian distribution)
        ranks = np.arange(POP_SIZE)
        probabilities = np.exp(-ranks / (POP_SIZE * 0.3)) # Exponential decay selection pressure
        probabilities /= probabilities.sum()
        
        parent_idx = np.random.choice(POP_SIZE, p=probabilities)
        parent = population[parent_idx]
        
        # Create a mutated child
        child = parent.mutate(sigma=0.1)
        
        # Replace the worst individual in the population with the new child
        population[-1] = child
        
    print("\n Optimization Complete!")
