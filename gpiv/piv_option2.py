import os
import sys


class Piv:

    def __init__(self, before_height, after_height, template_size, step_size, propagate):
        self.before_height = before_height
        self.after_height = after_height
        self.template_size = template_size
        self.step_size = step_size
        self.propagate = propagate

    @property
    def before_height(self):
        return self.__before_height    
    @before_height.setter
    def before_height(self, before_height):
        if os.path.isfile(before_height):
            self.__before_height = before_height
        else:
            print("Invalid 'before' file.")
            sys.exit()
    
    @property
    def after_height(self):
        return self.__after_height    
    @after_height.setter
    def after_height(self, after_height):
        if os.path.isfile(after_height):
            self.__after_height = after_height
        else:
            print("Invalid 'after' file.")
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


def is_positive_integer(n):
	try:
		val = int(n)
		if val < 1:
			return False
	except ValueError:
		return False
	return True