import argparse
import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path
from common import validate_file_path,safe_path_resolution

def parse_args():
    parser=argparse.ArgumentParser(description="Visualize training history")
    parser.add_argument("--hist",type=str,required=True,help="Path to training history .npz file")
    return parser.parse_args()

def main():
    args=parse_args()

    hist_path=args.hist

    try:
        hist_path=safe_path_resolution(hist_path)
        validate_file_path(hist_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the path was right")
        exit(1)
    
    data=np.load(hist_path)
    train_loss=data["train_loss"]
    val_loss=data["val_loss"]
    train_acc=data["train_acc"]
    val_acc=data["val_acc"]

    hist_file=Path(hist_path)
    model_name=hist_file.name.replace("_training_history.npz","")

    visualize_dir=Path("../visualize")/f"{model_name}_visualize"
    visualize_dir.mkdir(parents=True,exist_ok=True)

    epochs=range(1,len(train_loss)+1)
    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(15,10))
    fig.suptitle(f"Training Histroy - {model_name}",fontsize=16)

    ax1.plot(epochs,train_loss,"b-",label="Train Loss",linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True,alpha=0.3)
    ax1.legend()

    ax2.plot(epochs,val_loss,"r-",label="Validation Loss",linewidth=2)
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True,alpha=0.3)
    ax2.legend()
    
    ax3.plot(epochs,train_acc,"g-",label="Train Accuracy",linewidth=2)
    ax3.set_title("Training Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.grid(True,alpha=0.3)
    ax3.legend()

    ax4.plot(epochs,val_acc,"orange",label="Validation Accuracy",linewidth=2)
    ax4.set_title("Validation Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy")
    ax4.grid(True,alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    output_file=visualize_dir/f"{model_name}_training_history.png"
    plt.savefig(output_file,dpi=300,bbox_inches="tight")
    print(f"Visualization saved to: {output_file}")

    plt.figure(figsize=(12,8))
    
    plt.subplot(2,1,1)
    plt.plot(epochs,train_loss,"b-",label="Train Loss",linewidth=2)
    plt.plot(epochs,val_loss,"r-",label="Validation Loss",linewidth=2)
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True,alpha=0.3)
    
    plt.subplot(2,1,2)
    plt.plot(epochs,train_acc,"g-",label="Train Accuracy",linewidth=2)
    plt.plot(epochs,val_acc,"orange",label="Validation Accuracy",linewidth=2)
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True,alpha=0.3)
    
    plt.tight_layout()
    
    combined_file=visualize_dir/f"{model_name}_combined_history.png"
    plt.savefig(combined_file,dpi=300,bbox_inches="tight")
    print(f"Combined visualization saved to: {combined_file}")

if __name__=="__main__":
    main()
    pass