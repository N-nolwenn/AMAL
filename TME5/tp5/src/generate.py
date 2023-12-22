from textloader import  string2code, id2lettre, code2string
import math
import torch
import numpy as np
from torch.nn.functional import log_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  TODO:  Ce fichier contient les différentes fonction de génération
def generate_sequence(model, initial_input, eos_token, max_length=100, device='cuda'):
    model.eval()

    h = torch.zeros((1, 1, model.latent_dim)).squeeze(0).to(device)

    generated_sequence = [initial_input]

    with torch.no_grad():
        for _ in range(max_length):
            # Conversion de l'entrée courante en indice (sauf pour le premier caractère)
            if generated_sequence[-1] != initial_input:
                current_input = torch.tensor([lettre2id[generated_sequence[-1]]]).to(device)
            else:
                current_input = torch.tensor([0]).to(device)  # Utilisez l'indice 0 (ou tout autre indice approprié) pour <START>

            y, h = model(current_input, h)
            probs = torch.softmax(y, dim=-1)
            next_index = torch.multinomial(probs.view(-1), 1).item()
            next_char = id2lettre[next_index]
            generated_sequence.append(next_char)

            if next_char == eos_token:
                break

    return ''.join(generated_sequence)


def generate_beam(model, initial_input, eos_token, k=5, max_length=100, device='cuda'):
    model.eval()

    h = torch.zeros((1, 1, model.latent_dim)).squeeze(0).to(device)

    generated_sequence = [initial_input]

    with torch.no_grad():
        for _ in range(max_length):
            # Conversion de l'entrée courante en indice (sauf pour le premier caractère)
            if generated_sequence[-1] != initial_input:
                current_input = torch.tensor([lettre2id[generated_sequence[-1]]]).to(device)
            else:
                current_input = torch.tensor([0]).to(device)  # Utilisez l'indice 0 (ou tout autre indice approprié) pour <START>

            y, h = model(current_input, h)
            probs = torch.softmax(y, dim=-1)

            # Use top-k sampling for beam search
            topk_values, topk_indices = torch.topk(probs.view(-1), k)
            sampled_indices = torch.multinomial(topk_values, 1)
            next_index = topk_indices[sampled_indices].item()

            next_char = id2lettre[next_index]
            generated_sequence.append(next_char)

            if next_char == eos_token:
                break

    return ''.join(generated_sequence)

# p_nucleus

def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """  
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
        # Forward sur le modèle RNN et calcul des logits
        logits = decoder(h)
        # Application du softmax pour obtenir les probabilités
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        # Tri des probabilités par ordre décroissant
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        # Calcul de la masse de probabilité cumulée
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Sélection des indices dont la masse cumulée est inférieure à alpha
        selected_indices = sorted_indices[cumulative_probs <= alpha]
        # Création d'un tenseur avec les probabilités réduites à zéro sauf pour les indices sélectionnés
        nucleus_probs = torch.zeros_like(probabilities)
        nucleus_probs[:, selected_indices] = probabilities[:, selected_indices]

        return nucleus_probs

    return compute
