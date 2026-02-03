import argparse
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import TwoLayerNet
from utils import normalize_data
from common import validate_file_path,safe_path_resolution

def parse_args():
    parser=argparse.ArgumentParser(description="Predict using trained model")
    parser.add_argument("--data",type=str,required=True,help="Path to test data CSV file (required)")
    parser.add_argument("--model",type=str,required=True,help="Path to trained model .npz file (required)")
    parser.add_argument("--outdir",type=str,default="../predicts",help="Path to save predictions (default: ../predicts)")
    return parser.parse_args()

def load_model(model_path):

    safe_path_resolution(model_path)
    validate_file_path(model_path)
    
    data=np.load(model_path)
    
    W1=data["W1"]
    b1=data["b1"]
    W2=data["W2"]
    b2=data["b2"]
    model_name=data["model_name"] if "model_name" in data else "unknown_model"
    
    input_size=W1.shape[1]
    hidden_size=W1.shape[0]
    output_size=W2.shape[0]

    model=TwoLayerNet(input_size,hidden_size,output_size,model_name=model_name)

    model.W1=W1
    model.b1=b1
    model.W2=W2
    model.b2=b2

    return model
def main():
    args=parse_args()
    
    try:
        model=load_model(args.model)
        print(f"Model loaded: {model.get_model_name()}")
        
        test_path=safe_path_resolution(args.data)
        validate_file_path(test_path)
        
        print(f"Loading test data from: {test_path}")
        test_df=pd.read_csv(test_path)

        if test_df.shape[1]==785:
            test_images=test_df.iloc[:,1:].values
            test_labels=test_df.iloc[:,0].values
            has_test_labels=True
        else:
            test_images=test_df.values
            test_labels=None
            has_test_labels=False

        test_images=normalize_data(test_images,f"{os.path.basename(test_path)}")
        predictions=model.predict(test_images)
        
        print("Prediction done.")

        output_lines=[]
        output_lines.append(f"Model: {model.get_model_name()}")
        output_lines.append(f"Test data: {test_path}")
        output_lines.append(f"Number of sample: {len(predictions)}")

        if has_test_labels:
            accuracy=np.mean(predictions==test_labels)*100
            print(f"The accuracy on this data-set was {accuracy:.2f}%")
            output_lines.append(f"The accuracy on this data-set was {accuracy:.2f}%")

        print(f"\nPrediction would be saved as {model.model_name}_prediction.out")

        for i,pred in enumerate(predictions):
            if has_test_labels and i<len(test_labels):
                correct="Yes" if pred==test_labels[i] else "No"
                output_lines.append(f"Sample {i+1:6d}: Predicted={pred}, True={correct}")
            else:
                output_lines.append(f"Sample {i+1:6d}: Predicted={pred}")

        output_text="\n".join(output_lines)

        predicts_dir=Path(args.outdir)
        predicts_dir.mkdir(parents=True,exist_ok=True)
        predicts_filename=f"{model.get_model_name()}_predicts.out"

        output_file=os.path.join(predicts_dir,predicts_filename)
        output_file=os.path.normpath(output_file)
        with open(output_file,"w",encoding="utf-8") as f:
            f.write(output_text)
        print(f"\nPredictions saved to at {output_file}...")

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__=="__main__":
    main()
    pass