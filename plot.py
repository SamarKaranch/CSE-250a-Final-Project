import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

def plot_train_metrics(train_file, output_file, reward_window: int = 50):
    try:
        df = pd.read_csv(train_file)
        # Use episode number as the X-axis index
        df = df.set_index('episode')
    except FileNotFoundError:
        print(f"Error: Log file not found at '{train_file}'. Cannot plot.")
        return

    # 1. Calculate Moving Average for Reward
    df['reward_ma'] = df['reward'].rolling(window=reward_window, min_periods=1).mean()
    # 2. Epsilon is usually smooth enough, use raw epsilon as the main line since its decay should be visible.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
    
    # left plot: Raw Reward Data 
    ax_reward = axes[0]
    ax_reward.plot(df.index, df['reward'], 
                   label='Raw Reward', 
                   color='lightblue', 
                   alpha=0.4, 
                   linewidth=0.5)
    # left plot: smoothed line reward  
    ax_reward.plot(df.index, df['reward_ma'], 
                   label=f'Moving Avg ({reward_window} episodes)', 
                   color='darkgreen', 
                   linewidth=2)
    
    # custom y-axis
    max_ma = df['reward'].max()
    max_limit = max_ma * 1.10           # max limit 10% higher than the maximum moving average observed
    ax_reward.set_ylim(ymin=0, ymax=max_limit)
    ax_reward.set_xlabel('Episode Number', fontsize=12)

    ax_reward.set_title(f'DQN Agent Reward', fontsize=16)
    ax_reward.set_ylabel('Reward per Episode', fontsize=12)
    ax_reward.grid(axis='y', linestyle='--', alpha=0.7)
    ax_reward.legend(loc='upper left', fontsize=10) # Moved to top left as requested

    ## second plot 
    ax_reward_zoom = axes[1]
    ax_reward_zoom.plot(df.index, df['reward'], 
                   label='Raw Reward', 
                   color='lightblue', 
                   alpha=0.4, 
                   linewidth=0.5)  
    ax_reward_zoom.plot(df.index, df['reward_ma'], 
                   label=f'Moving Avg ({reward_window} episodes)', 
                   color='darkgreen', 
                   linewidth=2)
    
    # custom y-axis
    max_ma = df['reward_ma'].max()
    max_limit = max_ma * 1.10           # max limit 10% higher than the maximum moving average observed
    ax_reward_zoom.set_ylim(ymin=0, ymax=max_limit)
    ax_reward_zoom.set_xlabel('Episode Number', fontsize=12)

    ax_reward_zoom.set_title(f'DQN Agent Reward (zoomed)', fontsize=16)
    ax_reward_zoom.set_ylabel('Reward per Episode', fontsize=12)
    ax_reward.grid(axis='y', linestyle='--', alpha=0.7)
    ax_reward.legend(loc='upper left', fontsize=10) # Moved to top left as requested

    # Final Figure Adjustments
    fig.suptitle('DQN Training (24 stages) Reward', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make space for suptitle
    plt.savefig(output_file, dpi=300)

def plot_val_reward_only(log_file, output_file, reward_window: int = 50, zoom: bool = False):
    try:
        df = pd.read_csv(log_file)
        df = df.set_index('episode')
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file}'. Cannot plot.")
        return

    df['reward_ma'] = df['reward'].rolling(window=reward_window, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['reward'],         # Plot 1: Raw Reward Data
            label='Raw Reward', 
            color='lightblue', 
            alpha=0.4, 
            linewidth=0.5)
    ax.plot(df.index, df['reward_ma'],      # Plot 2: Smoothed Reward (Moving Average)
            label=f'Moving Avg ({reward_window} episodes)', 
            color='darkorange', 
            linewidth=3) # Slightly thicker line for focus

    # custom y-axis
    if zoom: # zoom into moving avg
        max_ma = df['reward_ma'].max()
        max_limit = max_ma * 1.10           # max limit 10% of the max moving avg 
        ax.set_ylim(ymin=0, ymax=max_limit)
    else: # don't zoom
        max_ma = df['reward'].max()
        max_limit = max_ma * 1.10          
        ax.set_ylim(ymin=0, ymax=max_limit)

    ax.set_title(f'DQN Validation (8 stages) Reward over Episodes', fontsize=18, fontweight='bold')
    ax.set_xlabel('Episode Number', fontsize=14)
    ax.set_ylabel('Reward per Episode', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()
    if zoom:
        output_file = output_file.strip(".png") + "_zoom.png"
    plt.savefig(output_file)


#### plot ONE RUN's reward earned each timestep WITH ACCUMULATED REWARD 
def plot_1_run_reward_over_time(run_file: str, output_path: str = None):
    try:
        df = pd.read_csv(run_file)
    except FileNotFoundError:
        print(f"Log file not found at '{run_file}'. Cannot plot.")
        return
    if df.empty:
        print(f"{run_file} Log file is empty.")
        return

    # takes first row to plot (change index if you want a different run)
    run_data = df.iloc[0]
    
    # Use ast.literal_eval to safely convert the rewards_over_t string back into a list
    try:
        step_rewards = ast.literal_eval(run_data['rewards_over_t'])
    except Exception as e:
        print(f"Error parsing 'rewards_over_t' string: {e}")
        return

    cum_rewards = np.cumsum(step_rewards)
    
    timesteps = np.arange(1, len(cum_rewards) + 1)
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # plots accumulated reward line
    ax.plot(timesteps, cum_rewards, 
            label='Cumulative Reward', 
            color='darkblue', 
            linewidth=2)

    # scatter plot for single points of reward 
    ax.scatter(timesteps, step_rewards, 
               s=15, # Size of the marker
               label='Instantaneous Reward Events', 
               color='red', 
               alpha=0.6) 
    
    # mark the final reward
    final_t = timesteps[-1]
    final_cum_reward = cum_rewards[-1]
    ax.scatter(final_t, final_cum_reward, 
               s=300,
               marker='*', 
               color='gold', 
               edgecolor='black',
               label='Level Completion Point', 
               zorder=4) 


    # Set titles and labels
    level = run_data['level']
    total_reward = run_data['total_reward']
    ax.set_title(f'Agent Reward Trajectory for {level} (Total Reward: {total_reward:.2f})', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Timestep (t)', fontsize=14)
    ax.set_ylabel('Reward Value', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # --- SAVING/DISPLAYING LOGIC ---
    if output_path:
        plt.savefig(output_path)
        print(f"✅ Timestep plot saved to {output_path}")
        plt.close(fig)
    else:
        plt.show()


#### plot ONE RUN's single rewards, NO CUMULATIVE REWARD
def plot_1_run_instantaneous_rewards(run_file: str, output_path: str = None):
    try:
        df = pd.read_csv(run_file)
    except FileNotFoundError:
        print(f"Error: Log file not found at '{run_file}'. Cannot plot.")
        return
    if df.empty:
        print("Error: Log file is empty.")
        return

    run_data = df.iloc[0]
    try:
        step_rewards = ast.literal_eval(run_data['rewards_over_t'])
    except Exception as e:
        print(f"Error parsing 'rewards_over_t' string: {e}")
        return

    timesteps = np.arange(1, len(step_rewards) + 1)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(timesteps, step_rewards, 
                s=20, # Slightly larger marker for focus
                label='Reward at timestep', 
                color='red', 
                alpha=0.7) 
    
    # Highlight final data point (last timestep)
    final_t = timesteps[-1]
    final_reward = step_rewards[-1]
    
    ax.scatter(final_t, final_reward, 
                s=200,      # bigger size 
                marker='*', # star marker
                color='gold', 
                edgecolor='black',
                label='Level Completed (Final Reward)', 
                zorder=3) # plots on top

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    level = run_data['level']
    ax.set_title(f'Instantaneous Rewards per Timestep for {level}', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Timestep (t)', fontsize=14)
    ax.set_ylabel('Instantaneous Reward Value', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    
    # Set y-axis limits to focus on the range of instantaneous rewards
    min_reward = min(step_rewards)
    max_reward = max(step_rewards)
    # Add a buffer of 10% to the max/min values for padding
    buffer = (max_reward - min_reward) * 0.1
    ax.set_ylim(min_reward - buffer, max_reward + buffer)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Instantaneous timestep plot saved to {output_path}")


## Plots multiple runs' Moving Average reward by learn_steps, capped at a specified learnstep_cap
def plot_multi_run_steps(run_paths: dict, learnstep_cap: int = 50000, reward_window: int = 50, out_file: str = None, log_file: str = 'train_log.csv'):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for model_run in run_paths:
        try:
            run_path = f"models/{model_run}/{log_file}"
            df = pd.read_csv(run_path)
        except FileNotFoundError:
            print(f"Log file not found at '/models/{model_run}/{log_file}'")
            continue

        # only use data up to timestep cap
        df_pruned = df[df['learn_step'] <= learnstep_cap].copy()
        # df_pruned = df[df['episode'] <= timestep_cap].copy()
        
        if df_pruned.empty:
            print(f"{model_run} data points doesn't reach {learnstep_cap} timesteps. Skipping.")
            continue

        # calc moving avg for comparison
        df_pruned['reward_ma'] = df_pruned['reward'].rolling(window=reward_window, min_periods=1).mean()
        
        # plot smoothed line
        # ax.plot(df_pruned['episode'], df_pruned['reward_ma'], 
        ax.plot(df_pruned['learn_step'], df_pruned['reward_ma'], 
                label=f'{model_run}', 
                linewidth=2)
        
        # Optional: Plot raw data faintly for insight, but MA comparison is cleaner
        # ax.plot(df_pruned['learn_step'], df_pruned['reward'], 
        #         color=ax.lines[-1].get_color(), 
        #         alpha=0.1)

    ax.set_title(f'Learning-rate: Training (24 levels) Rewards over Learn steps', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Learn Steps ({learnstep_cap:,} learn steps or ~48k episodes)', fontsize=14)
    ax.set_ylabel(f'Reward (Moving Average = {reward_window} steps)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    
    ax.set_xlim(left=0, right=learnstep_cap * 1.01)
    plt.tight_layout()
    
    if out_file:
        plt.savefig(out_file)
        print(f"Plot saved to {out_file}.")
        plt.close(fig)
    else:
        plt.show()


## Plots multiple runs' Moving Average reward over episodes, capped at a specified ep_cap.
def plot_multi_run_ep(run_paths: dict,  ep_cap: int = 50000, reward_window: int = 50, out_file: str = None, log_file: str = 'test.csv'):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for model_run in run_paths:
        try:
            run_path = f"models/{model_run}/{log_file}"
            df = pd.read_csv(run_path)
        except FileNotFoundError:
            print(f"Log file not found at '/models/{model_run}/{log_file}'")
            continue

        df_pruned = df[df['episode'] <= ep_cap].copy()
        if df_pruned.empty:
            print(f"{model_run} data points doesn't reach {ep_cap} timesteps. Skipping.")
            continue

        df_pruned['reward_ma'] = df_pruned['reward'].rolling(window=reward_window, min_periods=1).mean()
        # plot the smoothed line
        ax.plot(df_pruned['episode'], df_pruned['reward_ma'], 
                label=f'{model_run}', 
                linewidth=2)
        
        # Optional: Plot raw data faintly for insight, but MA comparison is cleaner
        # ax.plot(df_pruned['learn_step'], df_pruned['reward'], 
        #         color=ax.lines[-1].get_color(), 
        #         alpha=0.1)

    ax.set_title(f'Learning-rate: Validation (8 levels) Rewards over Training Episodes', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel(f'Episodes ({ep_cap:,} total)', fontsize=14)
    ax.set_ylabel(f'Reward (Moving Average = {reward_window} episodes)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    

    # X-axis capping
    ax.set_xlim(left=0, right=ep_cap * 1.01)
    plt.tight_layout()
    
    if out_file:
        plt.savefig(out_file)
        print(f"Plot saved to {out_file}.")
        plt.close(fig)
    else:
        plt.show()


def main():
    #### plot "validation" for one model
    # val_data = "models/model_v2/test.csv"
    # output_file = "models/model_v2/test_reward_plot.png"
    # plot_test_reward_only(val_data, output_file, reward_window=50, zoom=True)
    # plot_val_reward_only(val_data, output_file, reward_window=50, zoom=False)

    #### plot "train" for one model
    # train_file = "models/model_v2/train_log.csv"
    # output_file = "models/model_v2/train_plot.png"
    # plot_train_metrics(train_file, output_file, reward_window = 50)

    #### plot one run: reward over time
    # plot_1_run_reward_over_time(run_file="test_500/SuperMarioBros-2-2-v0.csv", output_path="plots/500ep_SMB-2-2-v0_cum.png")
    # plot_1_run_instantaneous_rewards("test_500/SuperMarioBros-2-2-v0.csv", output_path="plots/500ep_SMB-2-2-v0.png")

    #### plot multiple runs 
    run_paths = ['lr1e-4', 'lr25e-5', 'lr5e-4', 'lr1e-3']
    plot_multi_run_steps(run_paths=run_paths,  learnstep_cap=4200000, reward_window=1000, out_file='plots/train_lr.png', log_file= 'train_log.csv')
    # plot_multi_run_ep(run_paths=run_paths,  ep_cap=50000, reward_window=100, out_file='plots/val_lr.png', log_file= 'test.csv')

    # learn_step=4200000 is roughly [47859, 48710, 49320] episodes long for each model run 
    # so ima just say roughly 48k episodes

if __name__ == "__main__":
    main()
