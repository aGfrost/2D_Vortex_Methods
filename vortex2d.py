import cmath
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import copy

class Vortex(object):

    def __init__(self, location, circulation, fixed=False):
        '''
        Location should be a complex number
        '''
        self.location = location
        self.circulation = circulation
        self.fixed = fixed
        self.path = [location]
        self.step = complex(0,0)


    def potential(self, location):
        '''
        Returns a complex potential at a given location
        '''

        return (complex(0,-1)*(cmath.log(location - self.location))*(self.circulation)/(2*cmath.pi))

    def potential_grad(self, location):
        '''
        Returns a complex gradient of the potential function at a given location
        '''

        return (complex(0,-1)*self.circulation/(2*cmath.pi))*(1/(location - self.location))

    def get_velocity(self, location):

        return complex(self.potential_grad(location).real,(-1*(self.potential_grad(location).imag)))

    def get_krasny_velocity(self, location, delta):
        relative_vector = location - self.location
        mod = abs(relative_vector)
        conj = relative_vector.conjugate()
        modified_grad = (complex(0,-1)*self.circulation/(2*cmath.pi))*(conj)*(1/abs(mod**2 + delta**2))
        return complex(modified_grad.real,(-1*(modified_grad.imag)))


class Source(object):

    def __init__(self, location, strength, fixed=True):
        '''
        Location should be a complex number
        '''
        self.location = location
        self.strength = strength
        self.fixed = fixed
        self.fixed = fixed
        self.path = [location]
        self.step = complex(0,0)

    def potential(self, location):
        '''
        Returns a complex potential at a given location
        '''

        return ((cmath.log(location - self.location))*(self.strength)/(2*cmath.pi))

    def potential_grad(self, location):
        '''
        Returns a complex gradient of the potential function at a given location
        '''

        return (self.strength/(2*cmath.pi))*(1/(location - self.location))

    def get_velocity(self, location):

        return complex(self.potential_grad(location).real,(-1*(self.potential_grad(location).imag)))


class Freestream(object):

    def __init__(self, velocity, fixed=True):
        '''
        Velocity should be a complex number
        '''
        self.velocity = velocity
        self.fixed = fixed

    def potential(self, location):
        '''
        Returns a complex potential at a given location
        '''

        return (self.velocity*location)

    def potential_grad(self, location):
        '''
        Returns a complex gradient of the potential function at a given location
        '''

        return complex(self.velocity)

    def get_velocity(self, location):

        return complex(self.potential_grad(location).real,(-1*(self.potential_grad(location).imag)))



class Tracer(object):

    def __init__(self, location):
        '''
        Location should be a complex number
        '''
        self.location = location
        self.path = [location]
        self.step = complex(0,0)

class Setup(object):

    def __init__(self, sources=None, vortices=None, freestreams=None, tracers=None):
        '''
        each kwarg should be a list of respective object,
        tracers is a list containing initial coordinates of each tracer
        '''
        self.sources = sources
        self.vortices = vortices
        self.freestreams = freestreams
        self.tracers = tracers
        self.all = list()
        if self.vortices is not None:
            self.all += self.vortices
        if self.sources is not None:
            self.all += self.sources
        if self.freestreams is not None:
            self.all += self.freestreams


    def total_potential(self, x, y):
        @np.vectorize
        def get_potential(x,y):
            result = complex()
            for element in self.all:
                result = result + element.potential(complex(x,y))
            return result
        return get_potential(x,y)

    def total_potential_gradient(self, x, y):
        @np.vectorize
        def get_potential_grad(x,y):
            result = complex()
            for element in self.all:
                result = result + element.potential_grad(complex(x,y))
            return result
        return get_potential_grad(x,y)

    def get_velocity(self, location):
        x = location.real
        y = location.imag
        return complex(self.total_potential_gradient(x,y).real,(-1*(self.total_potential_gradient(x,y).imag)))

    def get_element_velocity(self, element):
        '''
        returns complex velocity of an element due to all the other elements
        '''
        temp_set = set(self.all)
        temp_set.remove(element)
        velocity = complex()
        for i in temp_set:
            velocity = velocity + i.get_velocity(element.location)
        return velocity

    def get_element_velocity_krasny(self, element, delta):
        '''
        returns complex velocity of an element due to all the other elements
        '''
        temp_set = set(self.all)
        temp_set.remove(element)
        velocity = complex()
        for i in temp_set:
            velocity = velocity + i.get_krasny_velocity(element.location, delta)
        return velocity


    def update_euler(self, time_step):

        #get steps for each element

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.step = self.get_element_velocity(vortex)*time_step

        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.step = self.get_velocity(tracer.location)*time_step

        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.step = self.get_element_velocity(source)*time_step

        #update the positions of each element

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.location = vortex.location + vortex.step
                    vortex.path.append(complex(vortex.location))


        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.location = source.location + source.step
                    source.path.append(complex(source.location))


        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.location = tracer.location + tracer.step
                tracer.path.append(complex(tracer.location))


    def update_RK2(self, time_step):

        time_step = float(time_step)

        #get steps for each element
        time_step = time_step/2

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.step = self.get_element_velocity(vortex)*time_step

        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.step = self.get_velocity(tracer.location)*time_step

        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.step = self.get_element_velocity(source)*time_step

        #update the positions of each element

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.location = vortex.location + vortex.step
                    vortex.path.append(complex(vortex.location))

        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.location = source.location + source.step
                    source.path.append(complex(source.location))

        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.location = tracer.location + tracer.step
                tracer.path.append(complex(tracer.location))

        #get steps for each element
        time_step = time_step*2

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.step = self.get_element_velocity(vortex)*time_step - vortex.step

        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.step = self.get_velocity(tracer.location)*time_step - tracer.step

        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.step = self.get_element_velocity(source)*time_step - source.step

        #update the positions of each element

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.location = vortex.location + vortex.step
                    vortex.path.append(complex(vortex.location))

        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.location = source.location + source.step
                    source.path.append(complex(source.location))

        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.location = tracer.location + tracer.step
                tracer.path.append(complex(tracer.location))

    def update_RK2_krasny(self, time_step,delta):

        time_step = float(time_step)

        #get steps for each element
        time_step = time_step/2

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.step = self.get_element_velocity_krasny(vortex, delta)*time_step

        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.step = self.get_velocity(tracer.location)*time_step

        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.step = self.get_element_velocity(source)*time_step

        #update the positions of each element

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.location = vortex.location + vortex.step
                    vortex.path.append(complex(vortex.location))

        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.location = source.location + source.step
                    source.path.append(complex(source.location))

        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.location = tracer.location + tracer.step
                tracer.path.append(complex(tracer.location))

        #get steps for each element
        time_step = time_step*2

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.step = self.get_element_velocity_krasny(vortex, delta)*time_step - vortex.step

        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.step = self.get_velocity(tracer.location)*time_step - tracer.step

        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.step = self.get_element_velocity(source)*time_step - source.step

        #update the positions of each element

        if self.vortices is not None:
            for vortex in self.vortices:
                if vortex.fixed is not True:
                    vortex.location = vortex.location + vortex.step
                    vortex.path.append(complex(vortex.location))

        if self.sources is not None:
            for source in self.sources:
                if source.fixed is not True:
                    source.location = source.location + source.step
                    source.path.append(complex(source.location))

        if self.tracers is not None:
            for tracer in self.tracers:
                tracer.location = tracer.location + tracer.step
                tracer.path.append(complex(tracer.location))
