from preprocessing import PreProcessing
import numpy as np

def main():
    dataset = PreProcessing("wine_dataset.txt")
    dataset.show()
    dataset.normalize(ignore_first_column=True)
    dataset.show_normalized()
    

if __name__ == "__main__":
    main()