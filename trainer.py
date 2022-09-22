import numpy as np
import torch
from utils import log_to_txt, Logger

class Trainer:
    def __init__(self, env, eval_env, agent, args):
        self.args = args

        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.logger = Logger(args)

        self.start_step = args.start_step
        self.update_after = args.update_after
        self.max_step = args.max_step
        self.batch_size = args.batch_size
        self.update_every = args.update_every

        self.eval_flag = args.eval_flag
        self.eval_episode = args.eval_episode
        self.eval_freq = args.eval_freq

        self.best_reward = 0
        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0
        self.finish_flag = False

        self.action_max = env.action_max

    def evaluate(self):
        # Evaluate process
        self.eval_num += 1
        reward_list = []

        for epi in range(self.eval_episode):
            epi_reward = 0
            state = self.eval_env.reset()

            done = False

            while not done:
                action = self.agent.get_action(state, evaluation=True)
                next_state, reward, done, _ = self.eval_env.step(self.action_max * action)
                epi_reward += reward
                state = next_state
            reward_list.append(epi_reward)

        if self.args.log:
            self.logger.log_eval(self.episode, self.total_step, reward_list)
        print("Eval  |  total_step {}  |  episode {}  |  Average Reward {:.2f}  |  Max reward: {:.2f}  |  "
              "Min reward: {:.2f}".format(self.total_step, self.episode, sum(reward_list)/len(reward_list),
                                          max(reward_list), min(reward_list), np.std(reward_list)))

        torch.save(self.agent.critic.state_dict(), self.logger.root_log+'critic.pt')
        torch.save(self.agent.target_critic.state_dict(), self.logger.root_log+'target_critic.pt')
        torch.save(self.agent.actor.state_dict(), self.logger.root_log+'actor.pt')
        if sum(reward_list)/len(reward_list) >  self.best_reward:
            torch.save(self.agent.critic.state_dict(), self.logger.root_log+'best_critic.pt')
            torch.save(self.agent.target_critic.state_dict(), self.logger.root_log+'best_target_critic.pt')
            torch.save(self.agent.actor.state_dict(), self.logger.root_log+'best_actor.pt')

    def run(self):
        # Train-process start.
        while not self.finish_flag:
            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0

            state = self.env.reset()

            done = False

            # Episode start.
            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.total_step >= self.start_step:
                    action = self.agent.get_action(state, evaluation=False)
                else:
                    action = self.env.random_sample()

                next_state, reward, done, true_done = self.env.step(self.action_max * action)
                self.episode_reward += reward
                self.agent.buffer.push(state, action, reward, next_state, true_done)

                state = next_state

                # Update parameters
                if self.agent.buffer.size >= self.batch_size and self.total_step >= self.update_after and \
                        self.total_step % self.update_every == 0:
                    total_actor_loss = 0
                    total_critic_loss = 0
                    total_log_alpha_loss = 0
                    for i in range(self.update_every * self.args.n_update):
                        critic_loss, actor_loss, log_alpha_loss = self.agent.train()
                        total_critic_loss += critic_loss
                        total_actor_loss += actor_loss
                        total_log_alpha_loss += log_alpha_loss

                    # Print loss.
                    if self.args.show_loss:
                        print("Loss  |  Actor loss {:.3f}  |  Critic loss {:.3f}  |  Log-alpha loss {:.3f}"
                              .format(total_actor_loss / self.update_every, total_critic_loss / self.update_every,
                                      total_log_alpha_loss / self.update_every))
                # Evaluation.
                if self.eval_flag and self.total_step % self.eval_freq == 0:
                    self.evaluate()

                # Raise finish_flag.
                if self.total_step == self.max_step:
                    self.finish_flag = True