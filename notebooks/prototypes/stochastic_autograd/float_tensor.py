import numpy as np

class FloatTensor(object):
    
    def __init__(self,data,autograd=False,keepgrads=False,creators=None, creation_op=None, id=None, is_stochastic = False):
        
        self.data = data
        self.autograd = autograd
        self.keepgrads = keepgrads
        self.grad = None
        if(id is None):
            self.id = np.random.randint(0,100000)
        else:
            self.id = id
        
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}
        self.is_stochastic = is_stochastic
    
    def __add__(self,other):
        if(self.autograd):
            out = FloatTensor(self.data + other.data, autograd=True, creators=[self,other], creation_op="add")
            self.children[out.id] = 0
            other.children[out.id] = 0
            return out
        else:
            return FloatTensor(self.data + other.data)
        
    
    def __mul__(self,other):
        if(self.autograd):
            out = FloatTensor(self.data * other.data, autograd=True, creators=[self,other], creation_op="mul")
            self.children[out.id] = 0
            other.children[out.id] = 0
            
            return out
        else:
            return FloatTensor(self.data * other.data)

    def __truediv__(self,other):
        if(self.autograd):
            out = FloatTensor(self.data / other.data, autograd=True, creators=[self,other], creation_op="div")
            self.children[out.id] = 0
            other.children[out.id] = 0
            return out
        return FloatTensor(self.data / other.data)    
    
    def __sub__(self,other):
        if(self.autograd):
            out = FloatTensor(self.data - other.data, autograd=True, creators=[self,other], creation_op="sub")
            self.children[out.id] = 0
            other.children[out.id] = 0
            
            return out

        return FloatTensor(self.data - other.data)   
    
    def __repr__(self):
        return self.data.__repr__()
    
    def __neg__(self):
        if(self.autograd):
            out = FloatTensor(-self.data, autograd=True, creators=[self], creation_op="neg")
            self.children[out.id] = 0
        return FloatTensor(-self.data)   
    
    def mm(self, x):
        if(self.autograd):
            out = FloatTensor(self.data.dot(x.data), autograd=True, creators=[self,x], creation_op="mm")
            self.children[out.id] = 0
            x.children[out.id] = 0
            return out
        return FloatTensor(self.data.dot(x.data))
    
    def transpose(self):
        return FloatTensor(self.data.transpose())
    
    def all_children_grads_accounted_for(self):
        for id,cnt in self.children.items():
            if(cnt == 0):
                return False
        return True
    
    def backward(self,grad=None, grad_origin=None):
        if(self.autograd):
            if(grad is None):
                grad = FloatTensor(np.ones_like(self.data))
            
            if(grad_origin is not None):
                if(self.children[grad_origin.id] > 0):
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] += 1
            
            if(self.grad is None):
                self.grad = grad
            else:
                self.grad += grad
            
            # grads must not have grads of their own
            assert grad.autograd == False
            
            # only continue backpropping if there's something to backprop into
            # only continue backpropping if all gradients (from children) are accounted for
            # override waiting for children if "backprop" was called on this variable directly
            if(self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None)):

                if(self.creation_op == "add"):
                    self.creators[0].backward(grad, self)
                    self.creators[1].backward(grad, self)

                if(self.creation_op == "mul"):
                    self.creators[0].backward(self.grad * self.creators[1], self)
                    self.creators[1].backward(self.grad * self.creators[0], self)

                if(self.creation_op == "div"):
                    self.creators[0].backward(self.grad / self.creators[1], self)
                    self.creators[1].backward(self.grad / self.creators[0], self)

                if(self.creation_op == "sub"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(-self.grad, self)

                if(self.creation_op == "neg"):
                    self.creators[0].backward(-self.grad,self)
                    
                if(self.creation_op == "mm"):
                    self.creators[0].backward(self.grad.mm(self.creators[1].transpose()))
                    self.creators[1].backward(self.grad.transpose().mm(self.creators[0]))
                    
                if(not self.keepgrads):
                    self.grad = None

        
