# Progetto di Ottimizzazione, Intelligenza Artificiale e Machine Learning
## Assegnazione dei compiti
**La seguente è la suddivisione indicativa dei compiti in quanto ogni parte del progetto sarà comunque seguita da tutti i membri del gruppo**
### 1. Analisi del Dataset & Annotazioni
**Responsabile: Dominici**

- Esaminare le immagini e verificare che le classi siano bilanciate (quante immagini per sasso, carta, forbici)

- Generare statistiche su distribuzione, dimensioni immagini, numero classi


### 2. Preprocessing & DataLoader personalizzato
**Responsabile: Dominici**

- Preprocessing delle immagini (resize, normalizzazione,...)

- Creare un DataLoader custom con PyTorch 

- Eventuale Data augmentation 


### 3. Modellazione rete
**Responsabile: Spitaleri**

- Progettare la rete neurale

- Costruire pipeline di training 


### 4. Training, Validazione, Salvataggi
**Responsabile: Seck**

- Gestire ciclo di training con logging delle metriche

- Monitoraggio con TensorBoard e scelta del modello migliore

### 5. Testing del modello addestrato
**Responsabile: Seck**
  
- Valutazione su immagini mai viste 

- Grafici metriche: Accuracy, Confusion Matrix, mAP 

- (opzionale) Visualizzare bounding box e classe predetta su immagini


# Relazione Progetto: Ottimizzazione, Intelligenza Artificiale e Machine Learning

## Introduzione

L’obiettivo del progetto è la realizzazione di un modello di visione artificiale per il riconoscimento di gesti della mano corrispondenti alle classi “sasso”, “carta” e “forbici”, tramite immagini statiche o un feed continuo da webcam. Il modello è pensato per essere sufficientemente leggero ed efficiente per l’integrazione in un’applicazione mobile, ad esempio per consentire all’utente di giocare a “Sasso, carta, forbici” contro un’intelligenza artificiale.

---

## Descrizione del Dataset e Preprocessing

Il dataset è organizzato in cartelle, una per ciascuna classe (“rock”, “paper”, “scissors”), sia per il training che per il test. Il caricamento delle immagini è gestito dalla classe `RPSDataset`, che si occupa di:
- Scansione delle cartelle e associazione delle etichette numeriche alle classi.
- Lettura delle immagini tramite OpenCV.
- Preprocessing con diverse strategie (prevalentemente conversione in scala di grigi e ridimensionamento a 50x50 pixel).
- Eventuale data augmentation tramite la classe `Augmentations`, che applica trasformazioni stocastiche come flip orizzontale, rotazioni, jitter di colore, affine, random erasing e normalizzazione.

Il preprocessing offre anche alternative per la segmentazione della mano tramite HSV o blob analysis, ma la pipeline finale adotta principalmente la versione in scala di grigi per robustezza e velocità.

```python
# Esempio di preprocessing robusto:
def preprocess(img, target_size=(50, 50)):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    gray = np.expand_dims(gray, axis=0)
    return gray
```

Durante il caricamento vengono scartate le immagini che non possono essere preprocessate correttamente, garantendo la qualità dei dati in ingresso.

---

## Architettura e Scelte Tecniche

### Modello

Il modello di riferimento è una CNN compatta, ottimizzata per input 50x50 monocanale:

```python
class CNN(nn.Module):
    def __init__(self, input_size=50, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

Varianti più leggere o con meno augmentations sono state testate negli script di `experiments` (es. per rapidità o test su dispositivi meno potenti).

### Pipeline di Training

- **Trainer**: Gestione del ciclo di training e validazione, con salvataggio automatico del modello migliore, scheduler del learning rate, early stopping e logging delle metriche.
- **Ottimizzazione**: Adam optimizer, CrossEntropyLoss (con anche label smoothing in alcune varianti).
- **Augmentation**: Trasformazioni casuali sul train set per migliorare la robustezza.
- **MixUp**: In uno degli esperimenti è stato implementato anche il MixUp per generare batch interpolati, riducendo l’overfitting.
- **Scheduler**: Riduzione del learning rate ogni 10 epoche.

---

## Risultati

- L’accuratezza del modello, valutata su immagini mai viste, viene monitorata ad ogni epoca e il modello con la migliore accuracy viene salvato.
- Il risultato tipico raggiunge una test accuracy superiore all’85-90% (a seconda della variante e del preprocessing).
- Vengono stampate le metriche di loss e accuracy durante il training. Non sono stati inclusi grafici nel codice, ma è semplice integrarli.
- L’inferenza in tempo reale da webcam funziona in modo fluido e reattivo, mostrando la predizione sull’immagine live.

---

## Riflessione sui Problemi e Soluzioni

- **Bilanciamento del Dataset**: Il dataset è stato analizzato per garantire che le classi fossero rappresentate in modo equo.
- **Robustezza del Preprocessing**: Sono stati provati diversi metodi (HSV, largest blob, mediapipe, scala di grigi). Quello in scala di grigi è risultato più robusto e generalizzabile.
- **Overfitting**: Contrastato con augmentation avanzata (random erasing, affine, jitter, mixup) e regularizzazione (dropout).
- **Early Stopping**: Introdotto per prevenire l’overfitting e ridurre i tempi di addestramento.
- **Test su Webcam**: Implementata una pipeline dedicata che permette di testare il modello “dal vivo” e visualizzare la maschera filtrata in parallelo al frame originale.

---

## Esempio di utilizzo del modello con webcam

Il file `webcam.py` permette di acquisire il video dalla webcam, preprocessare in tempo reale i frame, passare l’input alla rete neurale e visualizzare la predizione sovrapposta all’immagine.

## Caratteristiche del progetto
- **Codice modulare:** Il progetto è organizzato in moduli distinti per dataset, augmentations, preprocessing, modello, training, main entrypoint, inferenza live e script di esperimenti, facilitando la manutenzione e l’estendibilità.
- **Preprocessing e Data Augmentation:** Ampio uso di tecniche di preprocessing e augmentation per migliorare la robustezza.
- **Custom Dataset/DataLoader:** Utilizzo di una classe Dataset personalizzata compatibile con PyTorch.
- **Validazione, testing, early stopping:** La pipeline di training prevede validazione ad ogni epoca, salvataggio del modello migliore, fase di test separata e meccanismi di early stopping.
- **Sperimentazione:** Sono stati testati diversi modelli e configurazioni iperparametriche tramite script dedicati, e la pipeline consente di cambiare dataset facilmente.
- **Bilanciamento delle classi:** In alcune versioni avanzate viene calcolato e utilizzato un bilanciamento delle classi nella funzione di loss.
- **Salvataggio/caricamento modello:** Il trainer salva il modello migliore e permette il caricamento per l’inferenza da webcam.

---

## Riferimenti ai file chiave del progetto

- [dataset.py](https://github.com/AleBillo/ProgettoOIAML/blob/main/ml2/ml2/src/dataset.py)
- [augmentations.py](https://github.com/AleBillo/ProgettoOIAML/blob/main/ml2/ml2/src/augmentations.py)
- [preprocess.py](https://github.com/AleBillo/ProgettoOIAML/blob/main/ml2/ml2/src/preprocess.py)
- [model.py](https://github.com/AleBillo/ProgettoOIAML/blob/main/ml2/ml2/src/model.py)
- [trainer.py](https://github.com/AleBillo/ProgettoOIAML/blob/main/ml2/ml2/src/trainer.py)
- [main.py](https://github.com/AleBillo/ProgettoOIAML/blob/main/ml2/ml2/src/main.py)
- [webcam.py](https://github.com/AleBillo/ProgettoOIAML/blob/main/ml2/ml2/src/webcam.py)
- [experiments/](https://github.com/AleBillo/ProgettoOIAML/tree/main/ml2/ml2/src/experiments)

---

## Membri del gruppo

- Jacopo Maria Spitaleri
- Alessandro Dominici
- Seck Mactar Ibrahima
