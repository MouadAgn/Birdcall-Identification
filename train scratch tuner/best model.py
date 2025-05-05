import json
import pprint

import torch


def show_best_hyperparameters():
    """
    Charge le modèle sauvegardé et affiche les hyperparamètres optimaux
    utilisés dans le modèle final.
    """
    try:
        # Essaie d'abord de charger le meilleur modèle
        checkpoint = torch.load("best_bird_audio_classifier.pth", 
                               map_location=torch.device('cpu'))
        model_type = "best_bird_audio_classifier.pth"
    except FileNotFoundError:
        try:
            # Si le meilleur modèle n'est pas trouvé, essaie le modèle final
            checkpoint = torch.load("final_bird_audio_classifier.pth", 
                                   map_location=torch.device('cpu'))
            model_type = "final_bird_audio_classifier.pth"
        except FileNotFoundError:
            print("Aucun modèle sauvegardé n'a été trouvé.")
            return

    # Extraire les hyperparamètres
    if 'hyperparameters' in checkpoint:
        hyperparams = checkpoint['hyperparameters']
        
        print(f"\n===== Hyperparamètres du modèle ({model_type}) =====\n")
        
        # Utiliser pprint pour un affichage plus lisible
        pprint.pprint(hyperparams)
        
        # Sauvegarder les hyperparamètres dans un fichier JSON pour référence
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(hyperparams, f, indent=4)
        print("\nLes hyperparamètres ont été sauvegardés dans 'best_hyperparameters.json'")
        
        # Informations supplémentaires du modèle si disponibles
        if 'val_acc' in checkpoint:
            print(f"\nMeilleure précision de validation: {checkpoint['val_acc']:.2f}%")
        if 'epoch' in checkpoint:
            print(f"Obtenue à l'époque: {checkpoint['epoch']}")
        
        # Afficher le nombre de classes
        if 'classes' in checkpoint:
            print(f"\nNombre de classes: {len(checkpoint['classes'])}")
    else:
        print("Aucun hyperparamètre trouvé dans le modèle sauvegardé.")

if __name__ == "__main__":
    show_best_hyperparameters()