if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.train import main
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep

    # [Mod]Added a new config file for our project
    sweep = Sweep(RUN, Config).load("/scratch_net/biwidl216/rl_course_14/project/our_approach/decision-diffuser/code/analysis/carla_inv.jsonl")
    # sweep = Sweep(RUN, Config).load("/scratch_net/biwidl216/rl_course_14/project/our_approach/decision-diffuser/code/analysis/default_inv.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
