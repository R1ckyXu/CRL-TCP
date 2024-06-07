import numpy as np
import pickle
import torch

from Model import LSTMNet


# Experience Replay
class ExperienceReplay(object):
    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = []
        self.max_memory = max_memory
        self.discount = discount
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def remember(self, experience):
        self.memory.append(experience)

    # The sampling probability is unequal, with new experiences entering the experience replay buffer having a higher probability of being selected.
    def get_batch(self, batch_size=10):
        if len(self.memory) > self.max_memory:
            del self.memory[:len(self.memory) - self.max_memory]

        if batch_size < len(self.memory):
            timerank = torch.linspace(1, len(self.memory), len(self.memory)).to(self.device)

            p = timerank / torch.sum(timerank.float())

            batch_idx = torch.multinomial(p, num_samples=batch_size, replacement=False)

            batch = [self.memory[idx] for idx in batch_idx.cpu().numpy()]
        else:
            batch = self.memory

        return batch


class BaseAgent(object):

    def __init__(self, histlen):
        self.single_testcases = True
        self.train_mode = True
        self.histlen = histlen

    def get_action(self, s):
        return 0

    def get_all_actions(self, states):
        #  Returns list of actions for all states
        return [self.get_action(s) for s in states]

    def reward(self, reward):
        pass

    def save(self, filename):
        # Stores agent as pickled file
        pickle.dump(self, open(filename + '.p', 'wb'), 2)

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename + '.p', 'rb'))


class NetworkAgent(BaseAgent):

    def __init__(self, hidden_size, histlen, state_size, lr, lambda_reg):
        super(NetworkAgent, self).__init__(histlen=histlen)

        self.experience_length = 10000
        self.experience_batch_size = 1000  # The number of experiences drawn from the experience array each time.
        self.experience = ExperienceReplay(max_memory=self.experience_length)
        self.episode_history = []  # Execution history
        self.iteration_counter = 0
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.lambda_reg = lambda_reg

        # LSTM
        self.model = LSTMNet(self.state_size, self.hidden_size)

        self.name = self.model.name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print(f"Using {torch.cuda.device_count()} GPUs {torch.cuda.get_device_name(0)}")
            self.model.to(self.device)
        else:
            print("Using CPU")

        self.model_fit = False

        self.opt = torch.optim.Adam(lr=self.lr, params=self.model.parameters())
        self.cro_loss = torch.nn.MSELoss()

    def get_action(self, s):
        if self.model_fit:
            with torch.no_grad():
                inputs_tensor = torch.tensor(np.array(s).reshape(1, -1), dtype=torch.float32).to(self.device)

                outputs = self.model(inputs_tensor)
                # Obtain prediction results
                a = outputs[0][0].item()
        else:
            a = np.random.random()

        if self.train_mode:
            self.episode_history.append((s, a))

        return a

    def reward(self, rewards):

        if not self.train_mode:
            return

        try:
            x = float(rewards)
            rewards = [x] * len(self.episode_history)
        except:
            if len(rewards) < len(self.episode_history):
                raise Exception(f'Too few rewards {len(rewards)} {len(self.episode_history)}')

        self.iteration_counter += 1

        for ((state, action), reward) in zip(self.episode_history, rewards):
            self.experience.remember((state, reward))

        self.episode_history = []

        self.model_fit = True
        # Train the network every 5 cycles
        if self.iteration_counter == 1 or self.iteration_counter % 5 == 0:
            self.learn_from_experience()

    def learn_from_experience(self):
        experiences = self.experience.get_batch(self.experience_batch_size)

        x, y = zip(*experiences)
        labels = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32).squeeze(dim=-1).to(self.device)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        outputs = self.model(x)[:, 0]

        loss = self.cro_loss(outputs, labels)

        self.opt.zero_grad()

        # Calculate the L2 regularization term
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param) ** 2

        total_loss = loss + self.lambda_reg * l2_reg

        total_loss.backward()

        self.opt.step()
