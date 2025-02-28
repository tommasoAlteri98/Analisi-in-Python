from ml5imp import KMeansIrisApp  # type: ignore

def avvia_menu(app):
    while True:
        print("\nMenu:")
        print("Digita 1 = Genera Modello")
        print("Digita 2 = Analisi Descrittiva")
        print("Digita 3 = Genera grafico dei dati")
        print("Digita 4 = Risultati ClassificationReport")
        print("Digita 0 per interrompere il programma e uscire")
        scelta = input("Seleziona un'opzione: ")
        
        if scelta == "1":
            app.genera_modello()
        elif scelta == "2":
            app.analisi_descrittiva()
        elif scelta == "3":
            app.visualizza_grafico()
        elif scelta == "4":
            app.risultati_classification_report()
        elif scelta == "0":
            print("Chiusura del programma.")
            break
        else:
            print("Opzione non valida, riprova.")

if __name__ == "__main__":
    app = KMeansIrisApp()
    avvia_menu(app)