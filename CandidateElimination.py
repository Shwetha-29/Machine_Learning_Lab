
class CandidateElimination:
    def __init__(self, domains):
        self.domains = domains
        self.num_attr = len(domains)
        self.S = [['Φ'] * self.num_attr]
        self.G = [['?'] * self.num_attr]

    def is_consistent(self, h, x):
        return all(h[i] == '?' or h[i] == x[i] for i in range(self.num_attr))

    def more_general(self, h1, h2):
        return all(h1[i] == '?' or h1[i] == h2[i] for i in range(self.num_attr))

    def generalize_S(self, x):
        for i in range(self.num_attr):
            if self.S[0][i] == 'Φ':
                self.S[0][i] = x[i]
            elif self.S[0][i] != x[i]:
                self.S[0][i] = '?'

    def specialize_G(self, x):
        new_G = []
        for g in self.G:
            if self.is_consistent(g, x):
                for i in range(self.num_attr):
                    if g[i] == '?':
                        for value in self.domains[i]:
                            if value != x[i]:
                                h = g.copy()
                                h[i] = value
                                if self.is_consistent(h, self.S[0]):
                                    new_G.append(h)
            else:
                new_G.append(g)
        self.G = new_G

    def prune_G(self):
        self.G = [
            g for g in self.G
            if not any(self.more_general(g2, g) and g2 != g for g2 in self.G)
        ]

    def update(self, x, label):
        if label == '+':  
            self.G = [g for g in self.G if self.is_consistent(g, x)]
            self.generalize_S(x)
        else:  
            self.specialize_G(x)
        self.prune_G()

    def display(self, step):
        print("\nAfter Example", step)
        print("S =", self.S)
        print("G =", self.G)

data = [
    (['Circular','Large','Light','Smooth','Thick'], '+'),
    (['Circular','Large','Light','Irregular','Thick'], '+'),
    (['Oval','Large','Dark','Smooth','Thin'], '-'),
    (['Oval','Large','Light','Irregular','Thick'], '+')
]

domains = [
    ['Circular','Oval'],      
    ['Large'],                
    ['Light','Dark'],          
    ['Smooth','Irregular'],    
    ['Thick','Thin']          
]

ce = CandidateElimination(domains)

step = 0
for x, label in data:
    step += 1
    print("\nProcessing Example", step, ":", x, "->", label)
    ce.update(x, label)
    ce.display(step)

print("\nFINAL RESULT")
print("Final S:", ce.S)
print("Final G:", ce.G)
