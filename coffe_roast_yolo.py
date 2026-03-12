# Importação bibliotecas
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import pandas as pd

# Caminho dataset
dataset_path = "dataset_path"

# Carregando modelo YOLO
model = YOLO("yolov8n-cls.pt")

# Treinamento do modelo
results = model.train(
        data=dataset_path,
        epochs=15,
        imgsz=224,
        project="runs/classify",
        name="train_cafe",
        verbose=True
    )

# Avaliação do modelo no conjunto de teste
metrics = model.val(split='test')

# Métricas
print("\n===== RESULTADOS DO MODELO =====")
print(f"Acurácia Top-1: {metrics.top1*100:.2f}%")
print(f"Velocidade de inferência: {metrics.speed['inference']:.2f} ms por imagem")
print(f"Classes detectadas: {model.names}")

# Gráficos
results_dir = model.trainer.save_dir
results_file = os.path.join(results_dir, "results.csv")

if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        
        df.columns = df.columns.str.strip()

        plt.style.use('seaborn-v0_8-whitegrid') 

        # 1. Gráfico de Perda (Loss) - Treino vs Validação
        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["train/loss"], label="Perda de Treinamento", marker='o', linestyle='-')
        plt.plot(df["epoch"], df["val/loss"], label="Perda de Validação", marker='x', linestyle='--')
        
        plt.xlabel("Época", fontsize=12)
        plt.ylabel("Loss (Erro)", fontsize=12)
        plt.title("Evolução da Perda: Treino vs. Validação", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(results_dir, "grafico_loss.png"))
        plt.show()

        # 2. Gráfico de Acurácia (Validação)
        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["metrics/accuracy_top1"], label="Acurácia de Validação (Top-1)", color='green', marker='o')
        
        plt.xlabel("Época", fontsize=12)
        plt.ylabel("Acurácia (0-1)", fontsize=12)
        plt.title("Evolução da Acurácia durante o Treinamento", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(results_dir, "grafico_acuracia.png"))
        plt.show()

else:
        print("\nArquivo de resultados não encontrado.")