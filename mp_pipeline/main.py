"""
TRAINING PROCEDURE SHOULD BE AS FOLLOWS:
1) Play N amount of games to create the training set
    - Store the game state of each move
    - Store the probabilities for each move
    - Store the end result of the game (total moves or end score)
2) Sample x amount of positions from the last N*20 games
3) Retrain the network based on the saved states
    - A custom loss function is used which takes into account the move probabilities and
      predicted end result (total moves or end score)
4) Play y games and see whether the new neural network achieves better results
    - The new game results must be better by z percent so that
      the new neural network is chosen

For ALPHA GO ZERO the values were:
N = 25000
x = 2048
y = 400

NB! Currently only last N games are used for retraining and no evaluation is done.
"""
import multiprocessing as mp
from mp_pipeline.simulator import Simulator
from mp_pipeline.nn_active import NeuralNetActive
from mp_pipeline.nn import NeuralNetwork
from mp_pipeline.training_set import TrainingSet
from mp_pipeline.utils import print_consts
from mp_pipeline.utils import NBR_CONCURRENT_GAMES, MAX_GAMES, SAMPLE_AMOUNT


if __name__ == '__main__':
    train_with_simple_mcts = True  # if set, simulates games with simepl mcts (without model) for initial training

    print_consts()

    q_model = mp.Queue()
    q_model_train = mp.Queue()
    q_training_set = mp.Queue()
    q_games = {i: mp.Queue() for i in range(NBR_CONCURRENT_GAMES)}
    event_can_train = mp.Event()

    # start training set
    training_set = TrainingSet(q_training_set, q_model_train, event_can_train)
    training_set.start()

    if train_with_simple_mcts:
        nbr_games = 2*NBR_CONCURRENT_GAMES  # should be a multiple of NBR_OF_CONCURRENT_GAMES
        sample_amount = 2_000_000  # how many moves are used to train the model

        q_training_set.put((None, 'MAX_GAMES', nbr_games))
        q_training_set.put((None, 'SAMPLE_AMOUNT', sample_amount))

        # start games - without model, simply let them run till end
        games_per_sim = nbr_games // NBR_CONCURRENT_GAMES
        proc_games = [Simulator(idx, q_game, q_model, q_training_set, games_per_sim, False) for idx, q_game in q_games.items()]

        for p in proc_games:
            p.start()
        for p in proc_games:
            p.join()

        # after this we have generated all data for training set, just in case need to clean up
        # training set should also have sent all data to model queue, once NN starts, it simpy gets it
        q_training_set.put((None, 'MAX_GAMES', MAX_GAMES))
        q_training_set.put((None, 'SAMPLE_AMOUNT', SAMPLE_AMOUNT))
        q_training_set.put((None, 'CLEAR'))

    # start games
    proc_games = [Simulator(idx, q_game, q_model, q_training_set) for idx, q_game in q_games.items()]

    for p in proc_games:
        p.start()

    # start Neural Network
    proc_nn = NeuralNetActive(q_model, q_games, NeuralNetwork(), event_can_train, q_model_train)
    # proc_nn.start()
    proc_nn.run()

    for p in proc_games:
        p.join()
