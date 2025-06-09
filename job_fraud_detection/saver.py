import os
import pickle


class Saver:
    """
    This is a class for serializing objects into JSON.
    """
    save_directory = os.path.join(os.path.dirname(__file__), '..',
                                  'models')
    save_file = ''

    def __init__(self) -> None:
        """
        Initializes the class with a save directory and a save file. If not
        already initalized, make a new directory.
        """
        if not os.path.isdir(self.save_directory):
            os.makedirs(self.save_directory)

    def save(self, model, name) -> None:
        """
        Saves a dictionary into a pickle file.

        Args:
            model: Trained model.
        """
        self.save_file = name
        try:
            save_path = os.path.join(self.save_directory, self.save_file)
            with open(save_path, 'wb') as file:
                pickle.dump(model, file)
        except Exception as e:
            print(f"Model was not saved. {str(e)}.")

    def load(self, name=None) -> None:
        """
        Unpacks a pickle file.
        """
        try:
            if name:
                self.save_file = name
            save_path = os.path.join(self.save_directory, self.save_file)
            if not os.path.isfile(save_path):
                print("No saved model. Save first")
                return None
            with open(save_path, 'rb') as file:
                model = pickle.load(file)
            print('Model loaded succesfully!')
            return model
        except Exception as e:
            print(f'{str(e)}')
            return None
