from dataclasses import dataclass
import operator

import elfi
import GPy as gpy
import numpy as np

from harness.elfi.tasks import ELFIInferenceProblem, ModelAndDiscrepancy

from . import ops
from . import simulator as si



# # Set observed data and a fixed value for delta_2
# y0 = ops.get_SF_data(cluster_size_bound)
t_obs = 2
mean_obs_bounds = (0, 350)
t1_bound = 30
a1_bound = 40
cluster_size_bound = 80
warmup_bounds = (15, 300)
d2 = 5.95
epsilon = 0.01
R1_bound = 15
R2_bound = 0.5
t1_bound = 30
y0 = ops.get_SF_data(cluster_size_bound)

def constraint(burden, a2, d2, a1, d1, *args):
    # setup
    a1_bound=40
    mean_obs_bounds=(0, 350)
    # check
    mask1 = a1.squeeze() < a1_bound
    means = np.array(si.analytical_means(burden, a2, d2, a1, d1)).reshape(3, -1)
    mask2 = np.logical_and(means[2, :] > mean_obs_bounds[0], means[2, :] < mean_obs_bounds[1])
    return np.logical_and(mask1, mask2)

def simulator_with_constraint(*inputs, **kwargs):
    batch_size = kwargs.pop('batch_size', 1)
    # check constraint
    mask = constraint(*inputs)
    # run simulation with feasible parameters
    masked_inputs = (inputs[0][mask], inputs[1][mask], inputs[2], inputs[3][mask], inputs[4][mask], inputs[5], inputs[6], inputs[7])
    output = ops.simulator(*masked_inputs, **kwargs, batch_size=np.sum(mask))
    # initialise output
    ERROR = np.nan
    dtype = np.float16
    output_dtype = [('clusters', dtype, cluster_size_bound+1),
                    ('n_obs', dtype),
                    ('n_clusters', dtype),
                    ('largest', dtype),
                    ('obs_times', dtype, t_obs*12),
                    ('n_oversized', dtype),
                    ('n_c', dtype),
                    ('n_nc', dtype),
                    ('time', np.float16),]
    error_output = np.array((ERROR * np.ones(cluster_size_bound + 1), ERROR, ERROR, ERROR, ERROR * np.ones(t_obs*12), ERROR, ERROR, ERROR, ERROR), dtype=output_dtype)
    masked_output = np.array([error_output] * batch_size)
    # substitute feasible simulations
    masked_output[mask] = output
    return masked_output

@dataclass
class TB(ELFIInferenceProblem):

    @property
    def name(self) -> str:
        return "tb"

    def constraint(self, theta):
        R1, R2, burden, t1 = np.split(theta.reshape(-1, 4), 4, axis=1)  # note assumed parameter order
        d2 = 5.95
        a2 = R2 * d2
        d1 = ops.Rt_to_d(R1, t1)
        a1 = ops.Rt_to_a(R1, t1)
        return constraint(burden.squeeze(), a2.squeeze(), d2, a1.squeeze(), d1.squeeze())

    def build_model(self) -> ModelAndDiscrepancy:
        m = elfi.new_model()

        burden = elfi.Prior('normal', 200, 30)

        #joint = elfi.RandomVariable(ops.JointPrior, burden, mean_obs_bounds, t1_bound, a1_bound)

        R1 = elfi.Prior('uniform', 1 + epsilon, R1_bound - (1 + epsilon))

        R2 = elfi.Prior('uniform', epsilon, R2_bound - epsilon)

        t1 = elfi.Prior('uniform', epsilon, t1_bound - epsilon)

        # DummyPrior takes a marginal from the joint prior
        #R2 = elfi.Prior(ops.DummyPrior, joint, 0)
        #R1 = elfi.Prior(ops.DummyPrior, joint, 1)
        #t1 = elfi.Prior(ops.DummyPrior, joint, 2)

        # Turn the epidemiological parameters to rate parameters for the simulator
        a2 = elfi.Operation(operator.mul, R2, d2)
        a1 = elfi.Operation(ops.Rt_to_a, R1, t1)
        d1 = elfi.Operation(ops.Rt_to_d, R1, t1)

        # Add the simulator
        sim = elfi.Simulator(simulator_with_constraint, burden, a2, d2, a1, d1, 2, cluster_size_bound, warmup_bounds, observed=y0)

        # Summaries extracted from the simulator output
        clusters = elfi.Summary(ops.pick, sim, 'clusters')
        n_obs = elfi.Summary(ops.pick, sim, 'n_obs')
        n_clusters = elfi.Summary(ops.pick, sim, 'n_clusters')
        largest = elfi.Summary(ops.pick, sim, 'largest')
        obs_times = elfi.Summary(ops.pick, sim, 'obs_times')

        # Distance
        dist = elfi.Discrepancy(ops.distance, n_obs, n_clusters, largest, clusters, obs_times)
        log_dist = elfi.Operation(np.log, dist)
        return ModelAndDiscrepancy(m, log_dist)

    def build_target_model(self, model: elfi.ElfiModel):
        bounds = {'burden': (125, 275), 'R1': (1 + epsilon, R1_bound), 'R2': (epsilon, R2_bound), 't1': (epsilon, t1_bound)}
        span = [bounds[name][1] - bounds[name][0] for name in model.parameter_names]
        kernel = gpy.kern.RBF(input_dim=len(model.parameter_names), ARD=True)
        for ii in range(kernel.input_dim):
            kernel.lengthscale[[ii]] = span[ii] / 5
            kernel.lengthscale[[ii]].set_prior(gpy.priors.Gamma(2, 2 / kernel.lengthscale[ii]), warning=False)
            kernel.variance = 0.5**2
            kernel.variance.set_prior(gpy.priors.Gamma(2, 2 / kernel.variance), warning=False)
            mf = gpy.mappings.Constant(len(model.parameter_names), 1)
            mf.C = 5
            mf.C.set_prior(gpy.priors.Gamma(2, 2 / mf.C), warning=False)
        return elfi.GPyRegression(model.parameter_names, bounds=bounds, kernel=kernel, mean_function=mf)
