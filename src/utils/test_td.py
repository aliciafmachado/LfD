from src.utils.replay import ReplayMemory

def test_td(debug=False):
    gamma = 0.99
    trajectories = [{'state': [1, 2, 3, 4, 5, 6, 7],'action':[0, 1, 0, 2, 0, 0, 1], 
                    'next_state':[2, 3, 4, 5, 6, 7, None], 'reward':[0, 1, 0, 2, 1, 1, 2]}, 
                    {'state': [1, 2, 3, 4, 5, 6, 7],'action':[0, 1, 0, 2, 0, 0, 1], 
                    'next_state':[2, 3, 4, 5, 6, 7, None], 'reward':[0, 1, 0, 2, 1, 1, 2]}]

    replay = ReplayMemory(capacity=100000, initial_data=trajectories, n_td=4, gamma=gamma)

    true_tds = 2 * [(5, 0 + gamma * 1 + (gamma ** 2) * 0 + (gamma ** 3) * 2, 4), 
                    (6, 1 + gamma * 0 + (gamma ** 2) * 2 + (gamma ** 3) * 1, 4), 
                    (7, 0 + gamma * 2 + (gamma ** 2) * 1 + (gamma ** 3) * 1, 4), 
                    (7, 2 + (gamma ** 1) * 1 + (gamma ** 2) * 1 + (gamma ** 3) * 2, 4), 
                    (7, 1 + (gamma ** 1) * 1 + (gamma ** 2) * 2, 3), 
                    (7, 1 + (gamma ** 1) * 2, 2), 
                    (7, 2, 1)] 

    for i in range(len(true_tds)):

        true_td = true_tds[i]
        all_replay_tds = replay.get(i, mem='demo')
        replay_tds = all_replay_tds[-3:]

        if debug:
            print('True TD')
            print(true_td)
            print('Replay TD')
            print(all_replay_tds)
            print()

        assert(true_td[0] == replay_tds[0])
        assert(true_td[1] == replay_tds[1])
        assert(true_td[2] == replay_tds[2])

    print("All checks passed :).")


if __name__ == "__main__":
    test_td()