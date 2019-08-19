import os
import sys
import piv_functions


class Piv:

    def __init__(self, before_height_file, after_height_file, template_size, step_size):
        self.before_height_file = before_height_file
        self.after_height_file = after_height_file
        self.template_size = template_size
        self.step_size = step_size
        self.propagate = False
        self.output_base_name = ""
        self.before_uncertainty_file = None
        self.after_uncertainty_file = None

    @property
    def before_height_file(self):
        return self.__before_height_file    
    @before_height_file.setter
    def before_height_file(self, before_height_file):
        if os.path.isfile(before_height_file):
            self.__before_height_file = before_height_file
        else:
            print("Invalid 'before' height file.")
            sys.exit()
    
    @property
    def after_height_file(self):
        return self.__after_height_file    
    @after_height_file.setter
    def after_height_file(self, after_height_file):
        if os.path.isfile(after_height_file):
            self.__after_height_file = after_height_file
        else:
            print("Invalid 'after' height file.")
            sys.exit()
    
    @property
    def template_size(self):
        return self.__template_size    
    @template_size.setter
    def template_size(self, template_size):
        if is_positive_integer(template_size):
            self.__template_size = template_size
        else:
            print("Template size must a positive integer.")
            sys.exit()

    @property
    def step_size(self):
        return self.__step_size    
    @step_size.setter
    def step_size(self, step_size):
        if is_positive_integer(step_size):
            self.__step_size = step_size
        else:
            print("Step size must a positive integer.")
            sys.exit()

    @property
    def propagate(self):
        return self.__propagate    
    @propagate.setter
    def propagate(self, propagate):
        if isinstance(propagate, bool):
            self.__propagate = propagate
        else:
            print("Propagate option must be 'True' or 'False'.")
            sys.exit() 

    @property
    def before_uncertainty_file(self):
        return self.__before_uncertainty_file    
    @before_uncertainty_file.setter
    def before_uncertainty_file(self, before_uncertainty_file):
        if before_uncertainty_file is None or os.path.isfile(before_uncertainty_file):
            self.__before_uncertainty_file = before_uncertainty_file
        else:
            print("Invalid 'before' uncertainty file.")
            sys.exit()

    @property
    def after_uncertainty_file(self):
        return self.__after_uncertainty_file
    @after_uncertainty_file.setter
    def after_uncertainty_file(self, after_uncertainty_file):
        if after_uncertainty_file is None or os.path.isfile(after_uncertainty_file):
            self.__after_uncertainty_file = after_uncertainty_file
        else:
            print("Invalid 'after' uncertainty file.")
            sys.exit()
    
    def run(self):
        before_height, before_uncertainty, after_height, after_uncertainty, geo_transform = piv_functions.get_image_arrays(
            self.__before_height_file,
            self.__before_uncertainty_file,
            self.__after_height_file,
            self.__after_uncertainty_file,
            self.__propagate
        )

        if not self.__propagate:
            print('Computing PIV.')
            piv_functions.run_piv(
                before_height,
                [],
                after_height,
                [],
                geo_transform,
                self.__template_size,
                self.__step_size,                
                False,
                self.output_base_name)
        
            # # plot the displacement vectors on top of 'from' image       
            # show(True, False, True, False, True, 1, False, 1)



def is_positive_integer(n):
	try:
		val = int(n)
		if val < 1:
			return False
	except ValueError:
		return False
	return True