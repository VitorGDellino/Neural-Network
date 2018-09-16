from preprocessing import PreProcessing
import numpy as np

def main():
    dataset = PreProcessing("wine_dataset.txt")
    dataset.show()
    dataset.normalize(ignore_first_column=True)
    dataset.switch_first_last_column()
    dataset.show_normalized()
    dataset.show()
    

if __name__ == "__main__":
    main()