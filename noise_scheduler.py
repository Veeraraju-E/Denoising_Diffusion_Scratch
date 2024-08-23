import torch

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