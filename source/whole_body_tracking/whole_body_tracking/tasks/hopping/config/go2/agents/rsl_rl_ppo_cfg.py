from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from whole_body_tracking.tasks.hopping.symmetry import augment_go2_jump_symmetry


@configclass
class Go2HoppingFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 1
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 100
    experiment_name = "go2_jump"
    empirical_normalization = False
    clip_actions = 100.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=augment_go2_jump_symmetry,
            mirror_loss_coeff=1.0,
        ),
    )


@configclass
class Go2HoppingTrampolinePPORunnerCfg(Go2HoppingFlatPPORunnerCfg):
    experiment_name = "go2_jump_trampoline"

@configclass
class Go2HoppingTrampolineTrackingCfgPPORunnerCfg(Go2HoppingTrampolinePPORunnerCfg):
    experiment_name = "go2_jump_trampoline_tracking_cfg"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=augment_go2_jump_symmetry,
            mirror_loss_coeff=5.0,
        ),
    )

