import importlib
import re

from coqpit import Coqpit


def to_camel(text):
    text = text.capitalize()
    return re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)


def setup_model(config: Coqpit):
    """
    Load model directly from configuration
    """
    if "discriminator_model" in config and "generator_model" in config:
        my_model = importlib.import_module("vitts.components.vocoder.models.gan")
        my_model = getattr(my_model, "GAN")
    else:
        my_model = importlib.import_module("vitts.components.vocoder.models." + config.model.lower())
        if config.model.lower() == "wavernn":
            my_model = getattr(my_model, "WAVERNN")
        elif config.model.lower() == "gan":
            my_model = getattr(my_model, "GAN")
        elif config.model.lower() == "wavegrad":
            my_model = getattr(my_model, "WAVEGRAD")
        else:
            try:
                my_model = getattr(my_model, to_camel(config.model))
            except ModuleNotFoundError as e:
                raise ValueError(f" ! Model {config.model} not exist") from e

        print(" >>> Vocoder model: {}".format(config.model))
        return my_model.init_from_config(config)


def setup_generator(c):
    """TODO: use config object as arguments"""
    print(" > Generator Model: {}".format(c.generator_model))
    my_model = importlib.import_module("vitts.components.vocoder.models." + c.generator_model.lower())
    my_model = getattr(my_model, to_camel(c.generator_model))
    # this is to preserve the Wavernn class name (instead of Wavernn)
    if c.generator_model.lower() in "hifigan_generator":
        model = my_model(in_channels=c.audio["num_mels"], out_channels=1, **c.generator_model_params)
    elif c.generator_model.lower() in "melgan_generator":
        model = my_model(
            in_channels=c.audio["num_mels"],
            out_channels=1,
            proj_kernel=7,
            base_channels=512,
            upsample_factors=c.generator_model_params["upsample_factors"],
            res_kernel=3,
            num_res_blocks=c.generator_model_params["num_res_blocks"],
        )
    elif c.generator_model in "melgan_fb_generator":
        raise ValueError("melgan_fb_generator is now fullband_melgan_generator")
    elif c.generator_model.lower() in "multiband_melgan_generator":
        model = my_model(
            in_channels=c.audio["num_mels"],
            out_channels=4,
            proj_kernel=7,
            base_channels=384,
            upsample_factors=c.generator_model_params["upsample_factors"],
            res_kernel=3,
            num_res_blocks=c.generator_model_params["num_res_blocks"],
        )
    elif c.generator_model.lower() in "fullband_melgan_generator":
        model = my_model(
            in_channels=c.audio["num_mels"],
            out_channels=1,
            proj_kernel=7,
            base_channels=512,
            upsample_factors=c.generator_model_params["upsample_factors"],
            res_kernel=3,
            num_res_blocks=c.generator_model_params["num_res_blocks"],
        )
    elif c.generator_model.lower() in "parallel_wavegan_generator":
        model = my_model(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            num_res_blocks=c.generator_model_params["num_res_blocks"],
            stacks=c.generator_model_params["stacks"],
            res_channels=64,
            gate_channels=128,
            skip_channels=64,
            aux_channels=c.audio["num_mels"],
            dropout=0.0,
            bias=True,
            use_weight_norm=True,
            upsample_factors=c.generator_model_params["upsample_factors"],
        )
    elif c.generator_model.lower() in "univnet_generator":
        model = my_model(**c.generator_model_params)
    else:
        raise NotImplementedError(f"Model {c.generator_model} not implemented!")
    return model


def setup_discriminator(c):
    """TODO: use config object as arguments"""
    print(" > Discriminator Model: {}".format(c.discriminator_model))
    if "parallel_wavegan" in c.discriminator_model:
        # my_model = importlib.import_module("vitts.components.vocoder.models.parallel_wavegan_discriminator")
        raise "Not implement parallel wave gan."
    else:
        my_model = importlib.import_module("vitts.components.vocoder.models." + c.discriminator_model.lower())
    my_model = getattr(my_model, to_camel(c.discriminator_model.lower()))
    if c.discriminator_model in "hifigan_discriminator":
        model = my_model()
    if c.discriminator_model in "random_window_discriminator":
        model = my_model(
            cond_channels=c.audio["num_mels"],
            hop_length=c.audio["hop_length"],
            uncond_disc_donwsample_factors=c.discriminator_model_params["uncond_disc_donwsample_factors"],
            cond_disc_downsample_factors=c.discriminator_model_params["cond_disc_downsample_factors"],
            cond_disc_out_channels=c.discriminator_model_params["cond_disc_out_channels"],
            window_sizes=c.discriminator_model_params["window_sizes"],
        )
    if c.discriminator_model in "melgan_multiscale_discriminator":
        model = my_model(
            in_channels=1,
            out_channels=1,
            kernel_sizes=(5, 3),
            base_channels=c.discriminator_model_params["base_channels"],
            max_channels=c.discriminator_model_params["max_channels"],
            downsample_factors=c.discriminator_model_params["downsample_factors"],
        )
    if c.discriminator_model == "residual_parallel_wavegan_discriminator":
        model = my_model(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            num_layers=c.discriminator_model_params["num_layers"],
            stacks=c.discriminator_model_params["stacks"],
            res_channels=64,
            gate_channels=128,
            skip_channels=64,
            dropout=0.0,
            bias=True,
            nonlinear_activation="LeakyReLU",
            nonlinear_activation_params={"negative_slope": 0.2},
        )
    if c.discriminator_model == "parallel_wavegan_discriminator":
        model = my_model(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            num_layers=c.discriminator_model_params["num_layers"],
            conv_channels=64,
            dilation_factor=1,
            nonlinear_activation="LeakyReLU",
            nonlinear_activation_params={"negative_slope": 0.2},
            bias=True,
        )
    if c.discriminator_model == "univnet_discriminator":
        model = my_model()
    return model
