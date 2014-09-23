import numpy as np
import json
import collections

class BayesianNetwork:
    '''
    This library will allow the user to create a fully defined Bayesian Network using custom nodes and probabilities.  
    
    This library will allow the following functionality:
    1) Create, observe and change variables, including their names, possible values and marginal and conditional distributions as appropriate.
    2) Add, observe and remove nodes to a Bayesian network
    3) Allow ways to create, observe and alter the structure of the network
    4) Save and load a Bayesian network from file (I strongly recommend JSON)
    5) Set and remove hard evidence on multiple variables
    6) Do inference on the network 
    
    Please refer to module documentation for specific function calls and argument requirements
    
    '''
    def __init__(self,name):
        self.nodes = []
        self.links = []
        self.name = name
    
    def addNode(self, node):
        self.nodes.append(node)
    
    def addLink(self, parent, child):
        self.nodes[self.nodes.index(child)].addParent(parent)
        self.links.append([parent, child])
        
    def getNodes(self):
        return self.nodes
    
    def deleteNode(self, node):
        self.nodes.pop(self.nodes.index(node))
    
    def showNodes(self):
        for node in self.nodes:
            print('Name: ',node.getName(), '    States:', node.getStates(), '    Parents:', node.getParentNames())
    
    def showLinks(self):
        for link in self.links:
            print(link[0].getName(),"-->",link[1].getName())
            
    def clearEvidence(self):
        for node in self.nodes:
            node.setEvidenceStr()
            node.setPotential(node.getDistribution())
            restoreDimensions = []
            restoreDimensions.append(node.getName())
            for parent in node.getParents():
                restoreDimensions.append(parent.getName())
            node.setDimension(restoreDimensions)
        
    def getInference(self, infNode):
        nodeNames = []
        variableCount = len(self.nodes)
        for node in self.nodes:
            nodeNames.append(node.getName())
        nodeNames.sort()
        sortedNodes = []
        for names in nodeNames:
            for node in self.nodes:
                if names == node.getName():
                    sortedNodes.append(node)
        tileList = []
        
        #add new dimensions, tile, and swap
        for node in self.nodes:
            tmpPot = node.getPotential()
            if isinstance(node.getDimension(), str):
                dimensions = [node.getDimension()]
            else:
                dimensions = node.getDimension()[::-1]
            for i in range(0,variableCount):
                #check if dimensions of this node match the sorted node - dimension names are added in reverse order in comparison to distributions
                if i > len(dimensions) - 1:
                    dimensions.append('dummy')
                if dimensions[i] != sortedNodes[i].getName():
                #if they don't, check if the dimension is in the node, but in another axis
                    if sortedNodes[i].getName() in dimensions:
                        newIndex = dimensions.index(sortedNodes[i].getName())
                        tmpPot = np.swapaxes(tmpPot, i, newIndex)
                        dimensions[i], dimensions[newIndex] = dimensions[newIndex], dimensions[i]
                        tileList.append(1)
                    #if the dimension isn't in the node, add a new dimension for it
                    else:
                        tmpPot = np.expand_dims(tmpPot,axis = i)
                        tileList.append(len(sortedNodes[i].getStates()))
                        dimensions.insert(i,sortedNodes[i].getName())
                else:
                    tileList.append(1) 
            tmpPot = np.tile(tmpPot,tileList)
            node.setPotential(tmpPot)
            tileList = []
        
        #time to multiply distributions
        for node in self.nodes:
            if self.nodes.index(node) == 0:
                product = node.getPotential()
            else:
                product = product * node.getPotential()
        
        knownAxis = sortedNodes.index(infNode)
        
        axisTup = ()
        for i in range(0,variableCount):
            if i != knownAxis:
                tempTup = (i,)
                axisTup += tempTup
        
        unnormalizedTmp = np.sum(product,axis=axisTup) 
        marginal = unnormalizedTmp / np.sum(unnormalizedTmp)
        return marginal

    def jdefault(self,o):
        return o.__dict__
    
    def saveJSON(self):
       
        network_nodes = self.getNodes()
        
        lst = []

            
        for node in range(len(network_nodes)):
            d = collections.OrderedDict()
            #d = {}
            n={}
            #s={}
            parentName = []

            for y in network_nodes[node].getParents():
                parentName.append(y.getName())
                 
            d['Network Node'] = network_nodes[node].getName()
            d['States'] =  network_nodes[node].getStates()
            d['Parents'] = parentName
            d['Distribution'] = network_nodes[node].getDistribution().tolist()
            lst.append(d)
            #lst.append(n)
                #lst.append(s)
                #lst.append(values_lst)
        #for x in range(len(lst)):
        #    s.update(lst[x])
        lst.insert(0,{'Network Name':self.name})
        with open('bayesian_network.json', 'w') as f:
            json.dump(lst,f,indent =1 )
        #return json.dumps([{'Network Name:':'Monsu Network','variables':s}],sort_keys=True,indent = 1,separators=(',',']'))
        return json.dumps(lst,indent = 1)
        f.close()
        
    def loadJSON(self,file):
        json_file = file
        json_data = open(json_file)
        data = json.load(json_data)
        print(data)
        print(len(data))
        counter = 0
        for nodes in range(1,len(data)): #create nodes
            
            nodeName = data[nodes]['Network Node']
            nodestates = data[nodes]['States']
            self.addNode(Node(nodeName,nodestates))
        
        allNodes = self.getNodes()   
        for newnodes in range(1,len(data)):
    
            nodeParents = data[newnodes]['Parents']
            nodeDistribution = data[newnodes]['Distribution']
            allNodes[counter].setDistribution(nodeDistribution)
            for allParents in range(len(nodeParents)):
                for k in range(len(allNodes)):    
                    if nodeParents[allParents] == allNodes[k].getName():
                        nodeParent = allNodes[k]
                        
                self.addLink(nodeParent,allNodes[counter]) #adds as many parents as the child has
            
            counter = counter+1
        success = "Network Successful"
        return success 
        
class Node:
    '''
    The Node object is used to populate a Bayesian Network.  Its attributes are:
        1) Name
        2) States
        3) Parents
        4) Distribution
    
    The name and states are needed to initialize a node object.  Please refer to the documentation for additional functions.
    '''
    
    def __init__(self, name, states):
        self.name = name
        self.evidence = ""
        self.states = []
        self.parents = []
        self.distribution = []
        self.dimensions = []
        self.dimensions.append(name)
        self.potential = []
        for i in states:
            self.states.append(i)
    
    def getName(self):
        return self.name
    
    def getStates(self):
        return self.states
    
    def getParents(self):
        return self.parents
 
    def getParentNames(self):
        if len(self.parents) == 0:
            return 'None'
        else:
            par = []
            for parent in self.parents:
                par.append(parent.getName())
            return par
        
    def changeName(self, name):
        self.name = name
    
    def addParent(self, node):
        self.parents.append(node)
        self.dimensions.append(node.getName())
    
    def setDistribution(self,probs):
        self.distribution = np.array(probs,float)
        self.potential = self.distribution
          
    def getDistribution(self):
        return self.distribution;
    
    def getPotential(self):
        return self.potential
    
    def setPotential(self,probs):
        self.potential = np.array(probs,float)
    
    def getDimension(self):
        return self.dimensions
    
    def setDimension(self,dim):
        self.dimensions = dim

    def setEvidenceStr(self):
        self.evidence = ""
        
    def setEvidence(self,ev,bn):
        #first check yourself
        if ev in self.getStates():
            self.evidence = ev
            knownIndex = self.getStates().index(ev)
            if self.potential.ndim <= 1:
                self.potential = self.potential[knownIndex]
            elif self.potential.ndim == 2:
                self.potential = self.potential[:,knownIndex]
            elif self.potential.ndim == 3:
                self.potential = self.potential[:,:,knownIndex]
            else:
                self.potential = self.potential[:,:,:,knownIndex]
            #the first dimension in the dimensions list is always the node on which evidence is set
            self.dimensions.pop(0)
        
            #then check if the evidence node is a parent to the other nodes
            #handles only up to 2 parent nodes
            for node in bn.getNodes():
                if self in node.getParents():
                    thisDimension = node.getDimension().index(self.name)
                    childPotential = node.getPotential()
                    if childPotential.ndim == 1:
                        node.setPotential(childPotential[knownIndex])
                        newDims = node.getDimension()
                        newDims.pop(0)
                        node.setDimension(newDims)
                    elif childPotential.ndim == 2:
                        if thisDimension == 0:
                            node.setPotential(childPotential[:,knownIndex])
                            newDims = node.getDimension()
                            newDims.pop(0)
                            node.setDimension(newDims)
                        else:
                            node.setPotential(childPotential[knownIndex,:])
                            newDims = node.getDimension()
                            newDims.pop(1)
                            node.setDimension(newDims)
                    elif childPotential.ndim == 3:
                        if thisDimension == 0:
                            newDims = node.getDimension()
                            newDims.pop(2)
                            node.setDimension(newDims)
                        elif thisDimension == 1:
                            node.setPotential(childPotential[:,knownIndex,:])
                            newDims = node.getDimension()
                            newDims.pop(1)
                            node.setDimension(newDims)
                        else:
                            node.setPotential(childPotential[knownIndex,:,:])
                            newDims = node.getDimension()
                            newDims.pop(2)
                            node.setDimension(newDims)
