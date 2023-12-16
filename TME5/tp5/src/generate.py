from textloader import  string2code, id2lettre, code2string
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200, C = False):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    
    if start == "" :
        raise ValueError('Starting sequence cannot be empty.')
    else :
        # Embedding de la séquence de départ start
        x = string2code(start).to(device)
        x_emb = emb(x).to(device)
        
        # Forward sur le modèle RNN et calcul des états externe/interne
        h = torch.zeros(1, rnn.latent_dim, dtype = torch.float).to(device)
        
        if C:
            Ct = torch.zeros(1, rnn.latent_dim, dtype = torch.float).to(device)
            ht = rnn.forward(x_emb, h, Ct)[-1]
        else:
            ht = rnn.forward(x_emb, h)[-1]
        
        # Calcul du premier élément de la séquence
        distribution = torch.nn.LogSoftmax(decoder(ht) )
        output = torch.argmax(distribution, axis = 1)
        
        # Initialisation de la séquence générée et de sa taille
        gen_sequence = [output]
        gen_len = 1
        
        while gen_sequence[-1] != eos and len(gen_sequence) < maxlen:
            
            if C:
                ht, Ct = rnn.one_step(emb(gen_sequence[-1]), ht, Ct)
            else:
                ht = rnn.one_step(emb(gen_sequence[-1]), ht)
                
            distribution = torch.nn.LogSoftmax( decoder(ht) )
            output = torch.argmax(distribution, axis = 1)
            
            gen_sequence.append(output)
            
    return start + ' --> ' + start + code2string(gen_sequence)

def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search
    # Initialisation de la séquence générée et des scores associés
    sequences = [[string2code(start)], 0.0]
    gen_len = 0

    while gen_len < maxlen:
        new_sequences = []

        for sequence, score in sequences:
            # Récupérer le dernier élément de la séquence
            last_element = sequence[-1]
            
            # Embedding de la séquence
            x_emb = emb(torch.Tensor(last_element).long()).to(device)
            
            # Forward sur le modèle RNN et calcul des états externe/interne
            h = torch.zeros(1, rnn.latent_dim, dtype=torch.float).to(device)
            ht = rnn.forward(x_emb, h)[-1]
            
            # Calcul des distributions de probabilité sur les sorties
            distribution = torch.nn.LogSoftmax(rnn.decoder(ht)).squeeze()
            
            # Sélectionner les k symboles les plus probables
            top_k_indices = torch.topk(distribution, k=k).indices.cpu().numpy()
            
            for index in top_k_indices:
                new_sequence = sequence + [index]
                new_score = score + distribution[index].item()
                
                new_sequences.append([new_sequence, new_score])
                
        # Sélectionner les k meilleures séquences
        sequences = sorted(new_sequences, key=lambda x: x[1], reverse=True)[:k]
        
        # Vérifier si le symbole EOS est généré dans l'une des séquences
        for sequence, _ in sequences:
            if sequence[-1] == eos:
                return code2string(sequence)

        gen_len += 1
        
    # Sélectionner la séquence avec le score le plus élevé
    best_sequence = max(sequences, key=lambda x: x[1])[0]
    
    return code2string(best_sequence)


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
