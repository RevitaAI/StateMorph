# Problems

## Arguments
```java
public static final String NUM_PATHS_OPTION = "paths";
public static final String COMPENSATION_MATRIX_PATH = "comppath";
public static final String ALGORITHM_OPTION_COMPENSATE = "compensate";
```



## Cost function different from paper?

* Maybe misunderstanding
  * Paper
  * Code
* Would be better if walking through different costs

## Different prior?

Paper says 1, but code says 0.5

## Lexicon Encoding Cost



```java
protected double computeLexiconCost(Lexicon lexicon) {
    
    double lexCost = 0.0;
    for(int c = START_STATE + 1 ; c < END_STATE ; c++){
        lexCost += computePrequentialCostForClass(c);
    }
    return lexCost;        
}


private double computePrequentialCostForMap(Map<Byte, Integer> classCount) {
    Integer[] countsWrapper = new Integer[classCount.size()];
    countsWrapper = classCount.values().toArray(countsWrapper);
    double classLexCost = amorphous.math.AmorphousMath.computeCost(countsWrapper, constants.getPrior());
    return classLexCost;
}

// Another two layer of wrapping....

public static double computeCost(double[] counts, double prior){        
    if(counts.length == 0){
        return 0.0;
    }
    double cost = 0.0;
    double sumOfEvents = 0.0;
    double sumOfPriors = 0.0;
    for(double d: counts){
        cost -= log2Gamma(d + prior);
        cost += log2Gamma(prior);            
        sumOfEvents += d + prior;
        sumOfPriors += prior;
    }
    cost += log2Gamma(sumOfEvents);
    cost -= log2Gamma(sumOfPriors);
    return cost;
}

```


## Emmision cost when a morph is not in a state


```java
cost = - AmorphousMath.log2(constants.getPrior() / (s.classFrequency() + (s.classSize() + 1.0) * constants.getPrior()));
//System.out.println("\tpreq: cost of emission change: " + cost);
//System.out.println("\tpreq: cost of adding morph: " + costOfAddingMorphToClass(state, morphId));
cost += costOfAddingMorphToClass(state, morphId);
```
a
