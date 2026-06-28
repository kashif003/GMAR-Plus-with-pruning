import torch
class Head_importance:
    def __init__(self, model):
        self.model=model
        self.num_heads= model.config.num_attention_heads
        self.num_layers= model.config.num_hidden_layers
        self.hidden_size=model.config.hidden_size
        self.head_dim=int(self.hidden_size/self.num_heads)
        self.gradients= {}
        self.activations= {}
        self.activations_per_head= {}
        self.gradients_per_head= {}
        self.hook_handles= []
        self.gradient_score={}
        self.attention_score= {}

    def _get_gradients(self):
        return self.gradients
    
    def _get_activations(self):
        return self.activations

    def hook_value(self, name, activation=False):
        if activation:
            def hook(module, input, output):
                self.activations[name] = output[0]
        else:
            def hook(module, input, output):
                self.gradients[name] = output[0]
        return hook
   
    
    def register_forward_hooks(self):
        for i in range(self.num_layers):
            handle= self.model.vit.encoder.layer[i].attention.attention.register_forward_hook(self.hook_value(f"attention_{i+1}", activation=True))
            self.hook_handles.append(handle)

    def register_backward_hooks(self):
        for i in range(self.num_layers):
            handle= self.model.vit.encoder.layer[i].attention.attention.register_full_backward_hook(self.hook_value(f"attention_{i+1}"))
            self.hook_handles.append(handle)

    def get_attention_per_head(self):
        if not self.gradients:
            RuntimeError("activations are not computed or stored. First, create the self.activations dictionary.")
        for i in range(self.num_layers):
            batch, tokens, embed_dim =   self.activations[f"attention_{i+1}"].shape
            actt = self.activations[f"attention_{i+1}"].view(batch, tokens, self.num_heads, self.head_dim)
            head_actt = torch.unbind(actt, dim=2)
            self.activations_per_head[f"attention_{i+1}"]= list(head_actt)
        return self.activations_per_head
    
    def get_gradient_per_head(self):
        if not self.gradients:
            RuntimeError("Gradients are not computed or stored. First, create the self.gradients dictionary.")
        for i in range(self.num_layers):
            batch, tokens, embed_dim =   self.gradients[f"attention_{i+1}"].shape
            grads = self.gradients[f"attention_{i+1}"].view(batch, tokens, self.num_heads, self.head_dim)
            head_grads = torch.unbind(grads, dim=2)
            self.gradients_per_head[f"attention_{i+1}"]= list(head_grads)
        return self.gradients_per_head
    

    def get_geadient_score(self, norm="l1"):
        if not self.gradients_per_head:
            RuntimeError("Gradients per head are not computed or stored. First, create the self.gradients_per_head dictionary.")
        for i in range(self.num_layers):
            key= f"attention_{i+1}"
            self.gradient_score.setdefault(key,[])
            for j in range(self.num_heads):
                if norm=="l1":
                    score = self.gradients_per_head[key][j].abs().sum()
                    self.gradient_score[key].append(score.item())
                elif norm=="l2":
                    score = (self.gradients_per_head[key][j]**2).sum().sqrt()
                    self.gradient_score[key].append(score.item())
        return self.gradient_score
    
    def get_attention_score(self, norm="l1"):
        if not self.activations_per_head:
            RuntimeError("attention per head are not computed or stored. First, create the self.activations_per_head dictionary.")
        for i in range(self.num_layers):
            key= f"attention_{i+1}"
            self.attention_score.setdefault(key,[])
            for j in range(self.num_heads):
                if norm=="l1":
                    score = self.activations_per_head[key][j].abs().sum()
                    self.attention_score[key].append(score.item())
                elif norm=="l2":
                    score = (self.activations_per_head[key][j]**2).sum().sqrt()
                    self.attention_score[key].append(score.item())
        return self.attention_score


    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles= []

    def reset_all(self):
        self.gradients= {}
        self.activations= {}
        self.activations_per_head= {}
        self.gradients_per_head= {}
        self.hook_handles= []
        self.gradient_score={}
        self.attention_score= {}

