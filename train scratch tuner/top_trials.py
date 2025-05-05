import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib.cm import get_cmap


def plot_top_trials_with_threshold(study_name, db_path, similarity_threshold=5, top_n=10):
    """
    Create a horizontal bar chart showing the top N trials from an Optuna study,
    filtering out similar performing trials based on a threshold.
    
    Args:
        study_name (str): Name of the Optuna study
        db_path (str): Path to the SQLite database
        similarity_threshold (float): Minimum difference in accuracy to consider trials as distinct
        top_n (int): Number of top trials to display
    """
    # Load the study
    storage_name = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    # Get all completed trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # Sort trials by value (assuming higher is better)
    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)
    
    # Filter out similar trials
    filtered_trials = []
    for trial in sorted_trials:
        # If this is the first trial or if it's different enough from all filtered trials
        if not filtered_trials or all(abs(trial.value - t.value) >= similarity_threshold for t in filtered_trials):
            filtered_trials.append(trial)
        
        # Stop after collecting top_n trials
        if len(filtered_trials) >= top_n:
            break
    
    # Extract trial numbers and values
    trial_numbers = [f"#{t.number:04d}" for t in filtered_trials]
    values = [t.value for t in filtered_trials]
    
    # Create a horizontal bar chart
    plt.figure(figsize=(10, 7))
    
    # Use a colorful palette
    cmap = get_cmap('viridis')
    colors = [cmap(i/len(filtered_trials)) for i in range(len(filtered_trials))]
    
    # Create horizontal bars with reversed order (top performer at the top)
    y_pos = np.arange(len(trial_numbers))
    bars = plt.barh(y_pos, values, color=colors[::-1])
    
    # Add labels and values
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f"{width:.2f}%", 
                ha='left', va='center', fontsize=9)
    
    # Set chart properties
    plt.yticks(y_pos, trial_numbers[::-1])  # Reverse the order so best is on top
    plt.xlabel('Score')
    plt.title(f'Scores des Meilleurs Essais')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add padding to the right
    plt.xlim(0, max(values) * 1.15)
    
    # Ensure layout fits well
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('top_trials_scores_threshold.png', dpi=300)
    
    # Show the plot
    plt.show()
    
    # Print the details of the best trial
    best_trial = filtered_trials[0]
    print(f"\nMeilleur essai #{best_trial.number}: {best_trial.value:.2f}%")
    print("Paramètres:")
    for param_name, param_value in best_trial.params.items():
        print(f"    {param_name}: {param_value}")
    
    # Print info about the threshold
    print(f"\nNombre d'essais affichés: {len(filtered_trials)}")
    print(f"Seuil de similarité appliqué: {similarity_threshold}%")
    print("(Les essais avec une différence de score inférieure au seuil ont été filtrés)")

if __name__ == "__main__":
    # Parameters
    study_name = "bird_classifier_optimization"
    db_path = "optuna_bird_classifier.db"
    
    # Define threshold for considering trials as distinct
    # If two trials have accuracy difference less than this, only the better one will be shown
    similarity_threshold = 1.5  # in percentage points
    
    # Generate and display the plot
    plot_top_trials_with_threshold(
        study_name=study_name, 
        db_path=db_path, 
        similarity_threshold=similarity_threshold,
        top_n=10
    )
    
    print("Graphique sauvegardé sous 'top_trials_scores_threshold.png'")