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

L’obiettivo di questo progetto è la realizzazione di un sistema di visione artificiale, tramite tecniche di Deep Learning, per il riconoscimento dei gesti della mano corrispondenti alle classi “sasso”, “carta” e “forbici”. Il sistema è pensato per funzionare sia su immagini statiche sia in tempo reale da webcam, con particolare attenzione all’ottimizzazione del modello per velocità e robustezza.

---

## Configurazione e Struttura Modulare
Il progetto utilizza un file di configurazione (config/config.json) con il relativo schema (config/config_schema.json), in cui sono definiti i principali parametri del sistema come iper-parametri, percorsi ai dati, modalità di preprocessing, architettura della rete e impostazioni di training.
All’interno del codice, la cartella nets ospita diverse implementazioni di reti neurali, selezionabili tramite la configurazione. Le funzionalità di analisi statistica del dataset sono state raggruppate in una classe dedicata (dataset_analysis.py), con l’obiettivo di fornire output grafici e sintesi descrittive.
Un modulo separato (logger/) permette di configurare in modo flessibile TensorBoard per il logging, mentre la cartella dataset/ contiene i dati e le definizioni di eventuali trasformazioni o preprocessings utilizzabili.

## Descrizione del Dataset e Preprocessing
Il dataset utilizzato è strutturato in cartelle distinte per ciascuna classe (“rock”, “paper”, “scissors”) e per le suddivisioni training e test. L’analisi iniziale è stata fondamentale per verificare il bilanciamento tra le classi e la qualità delle immagini. A questo scopo sono stati generati grafici di distribuzione delle classi e preview di campioni casuali tramite il modulo `dataset_analysis.py`.

Le fasi di preprocessing sono state implementate per garantire uniformità e robustezza nel caricamento dei dati:
- Conversione delle immagini in scala di grigi e ridimensionamento a 50x50 pixel, scelta risultata la più efficace dopo test comparativi con segmentazioni HSV e blob analysis.
- Implementazione di un DataLoader custom (classe `RPSDataset` in `dataset.py`), in grado di scartare automaticamente immagini non processabili e mappare nomi delle classi in etichette numeriche.
- Data augmentation tramite la classe `Augmentations` (`dataset/augmentations.py`), applicando trasformazioni stocastiche come flip, rotazione, jitter di colore, affine e random erasing.

---

## Logging e Analisi
Una configurazione dedicata (logger/) permette di tracciare su TensorBoard l’andamento della loss e dell’accuracy in tempo reale, insieme a dati come architettura del modello e confusion matrix.
La classe di analisi del dataset in dataset_analysis.py offre funzioni per valutare la distribuzione delle classi e mostrare sample casuali, semplificando la generazione di report su bilanciamento e qualità del dataset.

Il progetto integra un sistema di logging delle metriche di training e validazione tramite TensorBoard. Questo è reso possibile grazie a un modulo dedicato che, durante l’addestramento, registra in automatico:
- Loss e accuratezza per ogni epoca
- Valori delle metriche su validation set
- Eventuali immagini di esempio, confusion matrix e altri dati diagnostici

In questo modo, è possibile:
- Monitorare l’andamento del training e identificare rapidamente episodi di overfitting, underfitting o plateaux delle metriche
- Confrontare tra loro diversi esperimenti in modo visivo
- Analizzare graficamente i trend delle metriche per prendere decisioni data-driven su modifiche di modello, preprocessing o iperparametri

---

## Architettura e Scelte Tecniche

## Configurazione e Struttura Modulare
Il progetto utilizza un file di configurazione (config/config.json) con il relativo schema (config/config_schema.json), in cui sono definiti i principali parametri del sistema come iper-parametri, percorsi ai dati, modalità di preprocessing, architettura della rete e impostazioni di training.
All’interno del codice, la cartella nets ospita diverse implementazioni di reti neurali, selezionabili tramite la configurazione. Le funzionalità di analisi statistica del dataset sono state raggruppate in una classe dedicata (dataset_analysis.py), con l’obiettivo di fornire output grafici e sintesi descrittive.
Un modulo separato (logger/) permette di configurare in modo flessibile TensorBoard per il logging, mentre la cartella dataset/ contiene i dati e le definizioni di eventuali trasformazioni o preprocessings utilizzabili.

## Pipeline di Training e Validazione
La classe Trainer gestisce l’addestramento e la validazione del modello. In particolare:
Viene istanziato un dataset di training (con preprocessing e augmentations) e un dataset di test, come descritto nel file di configurazione.
Vengono definiti, in modo centralizzato, ottimizzatore, funzione di loss e scheduler, tutti selezionabili via parametri nel file di configurazione.
La classe offre meccanismi come l’early stopping, il salvataggio del miglior modello e la gestione dei checkpoint, consentendo la ripresa del training.

---

## Risultati
- Accuratezza di test costantemente superiore all’85-90%, variabile in base alla pipeline di preprocessing e augmentation adottata.
- Salvataggio automatico del modello con la migliore performance su validation set.
- Metriche di loss e accuracy loggate durante il training, con possibilità di visualizzazione tramite TensorBoard.
- Inferenza in tempo reale via webcam, con predizione visualizzata “live” e possibilità di test immediato della robustezza del modello.

---

## Riflessione sui Problemi e Soluzioni

- **Bilanciamento delle classi:** Analisi statistica iniziale e, nelle versioni avanzate, ponderazione della loss per classi sbilanciate.
- **Preprocessing robusto:** Testate diverse pipeline (HSV, blob, MediaPipe, scala di grigi); scelta finale della scala di grigi per semplicità, efficienza e generalizzabilità.
- **Overfitting:** Affrontato con augmentation aggressiva e regolarizzazione (dropout, early stopping).
- **Testing su webcam:** Pipeline dedicata per validare la performance del modello in scenari reali e visualizzare anche le maschere di segmentazione.

---

## Quantizzazione e export
Il file `export.py` consente di esportare modelli in diversi formati (TorchScript, ONNX, custom), facilitando l’integrazione in applicazioni di produzione o dispositivi mobili applicando una quantizzazione ottimizzata per queste piattaforme.

---

## Caratteristiche del progetto
- **Codice modulare:** Suddiviso in moduli per dataset, augmentations, preprocessing, modello, training, inferenza, quantizzazione, export ed esperimenti.
- **Preprocessing e Data Augmentation:** Ampio uso di tecniche per aumentare la robustezza e la generalizzazione del modello.
- **Custom Dataset/DataLoader:** Gestione personalizzata compatibile con PyTorch.
- **Validazione, testing, early stopping:** Pipeline rigorosa con validazione ad ogni epoca e salvataggio del modello migliore.
- **Sperimentazione:** Facilmente configurabile tramite file di configurazione e script dedicati.
- **Bilanciamento delle classi:** Possibilità di bilanciare la loss per dataset non perfettamente bilanciati.
- **Salvataggio/caricamento modello:** Gestione automatica dei checkpoint.
- **Quantizzazione e export:** Strumenti dedicati per deployment su diversi dispositivi.

---

## Membri del gruppo

- Jacopo Maria Spitaleri
- Alessandro Dominici
- Seck Mactar Ibrahima
