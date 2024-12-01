import torch


# Noisy Image -> Denoising Diffusion Distribution -> Probabilistic Model -> Predicted Noise

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        """

        :param num_timesteps:
        :param beta_start:
        :param beta_end:
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # All the betas to linearly increase from start to end
        self.betas = torch.linspace(self.betas_start, self.betas_end, self.num_timesteps)

        # alpha_t = 1 - beta_t
        self.alphas = 1. - self.betas

        # alpha_bar => cumulative products of all alphas
        self.alpha_cumulative = torch.cumprod(self.alphas, dim=0)

        # we actually need sqrt of alpha_bar and 1 - alpha_bar
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1. - self.alpha_cumulative)

    def add_noise(self, original, noise, t):
        """

        :param original: original image [B, C, H, W]
        :param noise: the original noise sample
        :param t: which timestep? => decides exact amt of noise to add, based on derivation from original noise sample [B]
        one time step for every sample of the batch
        :return: Image with noise added
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        # reshape to make final size => [B, 1, 1, 1]
        sqrt_alpha_cumulative = self.sqrt_alpha_cumulative[t].reshape(batch_size)
        sqrt_one_minus_alpha_cumulative = self.sqrt_one_minus_alpha_cumulative[t].resize(batch_size)

        for i in range(len(original_shape) - 1):
            sqrt_alpha_cumulative = sqrt_alpha_cumulative.unsqueeze(-1)
            sqrt_one_minus_alpha_cumulative = sqrt_one_minus_alpha_cumulative.unsqueeze(-1)
            if i == 0:
                print(f"sqrt_alpha_cumulative.shape : {sqrt_alpha_cumulative.shape}, sqrt_one_minus_alpha_cumulative.shape : {sqrt_one_minus_alpha_cumulative.shape}")

        # we need to return sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t)*epsilon
        print(f"noise.shape : {noise.shape}, original.shape : {original.shape}")
        return sqrt_alpha_cumulative * original + sqrt_one_minus_alpha_cumulative * noise

    # now, we need a function to take in x_t and return a sampled datapoint from the reverse distribution
    def sample_from_reverse(self, x_t, noise_pred, t):
        """
        sampling based on previous time step
        :param x_t: x_t
        :param noise_pred: noise predicted from our models
        :param t: time step
        :return:
        """

        # from the reverse distribution, we can say that x_0 = (x_t - sqrt(1 - alpha_bar_t)*noise_pred)/(sqrt(alpha_bar_t))
        x_0 = (x_t - self.sqrt_one_minus_alpha_cumulative[t] * noise_pred) / (self.sqrt_alpha_cumulative[t])

        # imp!, don't forget to clamp x_0
        x_0 = torch.clamp(x_0, -1, 1.)

        print(f"in sample_from_reverse, x_0.shape : {x_0.shape}")
        # now, for the actual sampling, let's use mean and std given by
        # mu = (x_t - (1-alpha_t)*noise_pred/sqrt(1-alpha_bar_t))/(sqrt(alpha_bar_t))
        mu = (x_t - (self.betas[t]*noise_pred)/self.sqrt_one_minus_alpha_cumulative[t])/torch.sqrt(self.alphas[t])

        # also, we have a base case for t = 0
        if t == 0:
            return mu, x_0  # simply return mean (without noise)
        else:
            # now, we add noise after calculating var
            # var => same as variance conditioned on ground truth denoising distribution, given by (1-alpha_t)*(1-alpha_bar_(t-1))/(1-alpha_bar_t)
            var = self.betas[t] * self.alpha_cumulative[t-1] / (1 - self.alpha_cumulative[t])
            z = torch.randn(x_t.shape).to(x_t.device)

            # basically, we have to use the re-parameterization trick here (similar to VAEs)
            return mu + (var**0.5)*z, x_0

        # this completes Noise Scheduler!!