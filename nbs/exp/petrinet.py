
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/05_fastpm_petrinet.ipynb

class PetriNet:
    def __init__(self,P,T,I,O,m0):
        self.P,self.T,self.I,self.O,self.m0=P,T,I,O,m0
        self.Ic=O-I
        self.reset()
    def view(self):
        G = nx.DiGraph()
        G.graph['rankdir'] = 'LR'
        G.graph['dpi'] = 70



        for i in range(len(self.T)):
            G.add_node(self.T[i],shape='square',style='filled',fillcolor='grey',label=self.T[i])
        for i in range(len(self.P)):
            label=self.m[i] if self.m[i] else ' '
            if i==0: G.add_node(self.P[i],label=label,style='filled',fillcolor='green')
            elif i==len(self.P)-1: G.add_node(self.P[i],label=label,style='filled',fillcolor='red')
            else: G.add_node(self.P[i],label=label)

        for i in range(len(self.T)):
            for j in range(len(self.P)):
                #print(D_plus[i,j])
                if self.I[i,j]==1:
                    G.add_edge(self.P[j],self.T[i])
                #print(D_plus[i,j])
                if self.O[i,j]==1:
                    G.add_edge(self.T[i],self.P[j])


        return draw(G)
    def transition(self):
        t=(self.m>=self.I).all(axis=1)
        print(t)
        self.m=np.matmul(t, self.Ic)+self.m
        print(self.m)

    def reset(self):
        self.m=self.m0