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

## Missing features

### Word level simulated annealing and Hybrid simulated annealing

* Word-level: Random re-segmentation based on concave conves or linear
* Hybrid: Randomly choose morph-level simulated annealing or word-level simulated annealing

``` java

case CONCAVE:
    prob = (totalNumIteration/(round-totalNumIteration)+10.0)/9.0;
    break;
case CONVEX:
    if(round < 0.95 * totalNumIteration){
        prob = 1.0/(round + 1);
    }else{
        prob = 0.0;
    }
    break;
case LINEAR:
    prob = 1.0 - round/(0.95*totalNumIteration);
    break;

if(prob > 1.0){
    prob = 1.0;
}
if(prob < 0.0){
    prob = 0.0;
}
```

### Bulk de-registration

``` java
public static final String BULK_DEREGISTRATION_OPTION = "bulkdereg";
public static final String BULK_DEREG_PROBABILITY_OPTION = "bulkprob";
public static final String BULK_DEREG_STEM_UBOUND = "bulkubound";
public static final String BULK_DEREG_AFFIX_LBOUND = "bulklbound";

```

### Backword DP

