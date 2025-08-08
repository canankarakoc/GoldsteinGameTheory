package simulate;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;
import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.special.Erf;

/**
 *
 * @author rgoldst
 */
public class Simulate {
    // Constants
    double pi = Math.acos(-1.0);
    double sqrtpi = Math.sqrt(pi);
    double sqrt2 = Math.sqrt(2.0);
    double sqrt2pi = Math.sqrt(2.0*pi);
    double onePlusEpsilon = 1.0 + 1.0/999.0;
    
    // Simulation parameters
    // int burnInGen = 10000;
    int burnInGen = 0;
    int maxGen = 1001;
    boolean printSummary = false;
    boolean printMutations = true;
    boolean printDwellHisto = false;
    
    // Fitness parameters
    boolean acute = false;
    double nS = 0.01;
    double nV = 0.1 ;
    double beta = 1.0;
    double d0 = 0.1;
    double v0 = 0.5;
    double s0 = 0.0;
    double effectiveness = 1.0;
    double fitHostMax = -999.0;
    double fitPathMax = -999.0;
    
    // Mutation parameters
    double stdDevMove = 0.1;
    double stdDevAngle = 0.1 * pi;
    double probHostMutate = 0.1;
    int nSteps = 51;
    double[] possibleSteps = new double[nSteps];
    boolean fixedS = true;
    boolean fixedV = true;
    
    // Population parameters
    double hostPopSize = 1.0E4;
    double pathPopSize = 1.0E6;
 
    // Characterisation of results
    int histoPoints = 10;
    double[][] histo = new double[histoPoints][histoPoints];
    double totalHisto = 0.0;
    double totCount = 0.0;
    double fracNegSlope = 0.0;
    double noImmune = 0.0;
    double nTot = 0.0;
    int nDwell = 1000;
    double deltaDwell = 0.001;
    double[] dwellHisto = new double[nDwell];
    
    
    long seed = (long) 3248232;
    Random random = new Random();
    MersenneTwister twister = new MersenneTwister(seed);
    ExponentialDistribution exp;

    /**
     * @param args 
     */
    public static void main(String[] args) {
        Simulate simulate = new Simulate(args);
        simulate.run();
    }
    
    /**
     * Initialise with optional parameters
     * 
     * @param args 
     */
    Simulate(String[] args) {
        if (args.length > 2) {
            nV = Double.parseDouble(args[0]);
            nS = Double.parseDouble(args[1]);
            beta = Double.parseDouble(args[2]);
        }
        if (args.length > 4) {
            hostPopSize = Math.pow(10.0, Double.parseDouble(args[3]));
            pathPopSize = Math.pow(10.0, Double.parseDouble(args[4]));
        }
        // Define Gaussian set of values
        makeSteps();
        // Determine max fitness values
        findMaxFit();
    }
    
    void run() {
        
        // Initialise to random values and angles
        double s = random.nextDouble();
        double v = random.nextDouble();
        double sAngle = 0.0;
        if (!fixedS) {
            sAngle = random.nextDouble() * pi;
        }
        double vAngle = 0.0;
        if (!fixedV) {
            vAngle = random.nextDouble() * pi;
        }
        
        // Convert into slopes, intercepts, fitness
        double mS = Math.tan(sAngle);                       // Slopes
        double mV = Math.tan(vAngle);
        double bS = s - mS * v;                             // Intercepts
        double bV = v - mV * s;
        double hostFitness = computeHostFitness(v, s);      // Fitnesses
        double pathFitness = computePathFitness(v, s);
        
        double omegaHost = 0.0;
        double omegaPath = 0.0;
        double time = 0.0;
        double sFactor = 1.0;
        double vFactor = 1.0;
        
        if (fixedS) {
            sFactor = possibleSteps.length;
        }
        if (fixedV) {
            vFactor = possibleSteps.length;
        }
        
        // Storage for cumulative probabilities
        TreeMap<Double, double[]> cumProbHostMutMap = new TreeMap<>();
        TreeMap<Double, double[]> cumProbPathMutMap = new TreeMap<>();
   
        // Simulate
        for (int iGen = -burnInGen; iGen < maxGen; iGen++) {
            // Initialise cumulative values
            cumProbHostMutMap.clear();
            cumProbPathMutMap.clear();
            double cumProbHostMut = 0.0;
            double cumProbPathMut = 0.0;
            double cumProbHostNeut = 0.0;
            double cumProbPathNeut = 0.0;
            
            // Consider changes in host
            // Possible changes in s value
            for (int iSStep = 0; iSStep < possibleSteps.length; iSStep++) {
                double newS = s + stdDevMove * possibleSteps[iSStep];                           
                // Consider changes in angle
                int sMaxAngle = 1;
                if (!fixedS) {
                    sMaxAngle = possibleSteps.length;
                }
                for (int iSAngleStep = 0; iSAngleStep < sMaxAngle; iSAngleStep++) {
                    // Angle between 0 and pi
                    double newSAngle = 0.0;
                    if (!fixedS) {
                        newSAngle = (sAngle + stdDevAngle * possibleSteps[iSAngleStep] + 2.*pi)%pi;
                    }
                    double newMS = Math.tan(newSAngle);
                    cumProbHostNeut += sFactor;
                    // Find new intercept by rotating around current value of v
                    double newBS = newS - newMS * v;
                    // Find new intersection point
                    HashMap<String, double[]> newPoints = findPoints(bV, mV, newBS, newMS);
                    if (newPoints.size() > 0) {
                        // Compute new host fitness and prob acceptance
                        double fitMax = -1000.0;
                        double[] bestSV = new double[2];
                        for (double[] sv : newPoints.values()) {
                            double newHostFitness = computeHostFitness(sv[0], sv[1]);
                            if (newHostFitness > fitMax) {
                                bestSV[0] = sv[0];
                                bestSV[1] = sv[1];
                                fitMax = newHostFitness;
                            }
                        }      
                        double probAcceptance = computeProbAcceptance(hostFitness, fitMax, hostPopSize);
                        if (probAcceptance > 1.0E-4) {
                            // Save possible state
                            double[] state = new double[7];
                            state[0] = newBS;
                            state[1] = newSAngle;
                            state[2] = newMS;
                            state[3] = bestSV[0];
                            state[4] = bestSV[1];
                            state[5] = fitMax;
                            state[6] = probAcceptance;
                            cumProbHostMut += probAcceptance * sFactor;
                            cumProbHostMutMap.put(cumProbHostMut, state);
                        }
                    }
                }
            }
            
            // Consider changes in path
            // Possible changes in v value
            for (int iVStep = 0; iVStep < possibleSteps.length; iVStep++) {
                double newV = v + stdDevMove * possibleSteps[iVStep];
                int vMaxAngle = 1;
                if (!fixedV) {
                    vMaxAngle = possibleSteps.length;
                }
                for (int iVAngleStep = 0; iVAngleStep < vMaxAngle; iVAngleStep++) {
                    double newVAngle = 0.0;
                    if (!fixedV) {
                        newVAngle = (vAngle + stdDevAngle * possibleSteps[iVAngleStep]+2.0*pi)%pi;
                    }
                    double newMV = Math.tan(newVAngle);
                    cumProbPathNeut += vFactor;
                    double newBV = newV - newMV * s;
                    // Find new intersection point
                    HashMap<String, double[]> newPoints = findPoints(newBV, newMV, bS, mS);
                    if (newPoints.size() > 0) {
                        // Compute new host fitness and prob acceptance
                        double fitMax = -1000.0;
                        double[] bestSV = new double[2];
                        for (double[] sv : newPoints.values()) {
                            double newPathFitness = computePathFitness(sv[0], sv[1]);
                            if (newPathFitness > fitMax) {
                                bestSV[0] = sv[0];
                                bestSV[1] = sv[1];
                                fitMax = newPathFitness;
                            }
                        }                         // Find new intersection point
                        double probAcceptance = computeProbAcceptance(pathFitness, fitMax, pathPopSize);
                        if (probAcceptance > 1.0E-4) {
                            double[] state = new double[7];
                            state[0] = newBV;
                            state[1] = newVAngle;
                            state[2] = newMV;
                            state[3] = bestSV[0];
                            state[4] = bestSV[1];
                            state[5] = fitMax;
                            state[6] = probAcceptance;
                            cumProbPathMut += probAcceptance * vFactor;
                            cumProbPathMutMap.put(cumProbPathMut, state);
                        }
                    }
                }
            }

            double totalRate = probHostMutate * cumProbHostMut + (1.0 - probHostMutate) * cumProbPathMut;
            double probChooseHost = (probHostMutate * cumProbHostMut) / totalRate;
            boolean mutateHost = random.nextDouble() < probChooseHost;
            String label = "Path";
            if (mutateHost) label = "Host";
            if (iGen > 0) {
                exp = new ExponentialDistribution(twister, 1.0/totalRate);
                double dwell = exp.sample();
                totalHisto += dwell;
                int sBin = Math.round(Math.round(Math.floor(s*histoPoints)));
                int vBin = Math.round(Math.round(Math.floor(v*histoPoints)));
                sBin = Math.max(0, Math.min(histoPoints-1, sBin));
                vBin = Math.max(0, Math.min(histoPoints-1, vBin));
                histo[vBin][sBin] += dwell;
                int iDwell = Math.round(Math.round((dwell/deltaDwell)-0.5));
                if (iDwell >= 0 && iDwell < nDwell) {
                    dwellHisto[iDwell]++;
                }
                time += dwell;
                if (printMutations) {
                    System.out.format("xxx\t%d\t%.8f\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t\t%.5f\t%.5f\t%.5f\t%.5f\t\t%.5f\t%.5f\t\t%.5g\t%.5g\n", 
                            iGen, time, label, v, bV, vAngle, mV, s, bS, sAngle, mS,
                            pathFitness/fitPathMax, hostFitness/fitHostMax, omegaPath, omegaHost);
                }
            }
            if (mutateHost) {
                double newKey = cumProbHostMutMap.ceilingKey(cumProbHostMut*random.nextDouble());
                double[] state = cumProbHostMutMap.get(newKey);
                bS = state[0];
                sAngle = state[1];
                mS = state[2];
                v = state[3];
                s = state[4];
                hostFitness = state[5];
                pathFitness = computePathFitness(v, s);
            } else {
                double newKey = cumProbPathMutMap.ceilingKey(cumProbPathMut*random.nextDouble());               
                double[] state = cumProbPathMutMap.get(newKey);
                bV = state[0];
                vAngle = state[1];
                mV = state[2];
                v = state[3];
                s = state[4];
                pathFitness = state[5];
                hostFitness = computeHostFitness(v, s);
            }
            omegaHost = cumProbHostMut/cumProbHostNeut;
            omegaPath = cumProbPathMut/cumProbPathNeut;
            
            if (printMutations && iGen >= 0) {
                System.out.format("yyy\t%d\t%.8f\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t\t%.5f\t%.5f\t%.5f\t%.5f\t\t%.5f\t%.5f\t\t%.5g\t%.5g\n", 
                        iGen, time, label, v, bV, vAngle, mV, s, bS, sAngle, mS,
                        pathFitness/fitPathMax, hostFitness/fitHostMax, 
                        omegaPath, omegaHost);
            }
            
            if (false && iGen >= 0 && iGen%1000 == 0) {
                System.out.format("%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", 
                        iGen, s, bS, mS, hostFitness, v, bV, mV, pathFitness);
                if (mV < 0.0) {
                    fracNegSlope++;
                }
                if (s < 0.01) {
                    noImmune++;
                }
                nTot++;
            }
        }
        if (printSummary) {
            System.out.format("nS,nV,beta,hostPop,pathPop:\t%.4f\t%.4f\t%.4f\t%.2g\t%.2g\n", nS, nV, beta, hostPopSize, pathPopSize);
            System.out.format("Maxgen:\t%d\n", maxGen);
            System.out.format("Frac neg slope:\t%.4f\n",
                    (fracNegSlope/nTot));
            System.out.format("No immunity:\t%.4f\n",
                    (noImmune/nTot));
        }
        
        for (int iS = histoPoints-1; iS >= 0; iS--){
            for (int iV = 0; iV < histoPoints; iV++) {
                System.out.format("\t%.4f", histo[iV][iS]/totalHisto);
            }
            System.out.println();
        }        
        if (printDwellHisto) {
            for (int iDwell = 0; iDwell < nDwell; iDwell++) {
                System.out.format("%d\t%.8f\t%.8f\n", iDwell, (iDwell+0.5)*deltaDwell, dwellHisto[iDwell]);
            }
        }
    }
    
    double computeHostFitness(double v, double s) {
        double central = (v-0.5)*(v-0.5)+(s-0.5)*(s-0.5);
        if (true) return 1.0 + 0.0001 * Math.exp(-central/0.5);
        if (acute) {
            return s / (s + computeMortality(v, s));  
        }
        return 1.0/computeMortality(v, s);
    }

    double computePathFitness(double v, double s) {
        double central = (v-0.5)*(v-0.5)+(s-0.5)*(s-0.5);
        if (true) return 1.0 + 0.00001 * Math.exp(-central/0.5);
        if (acute) {
            return computeTransmission(v, s)/(s + computeMortality(v, s));
        }
        return computeTransmission(v, s)/computeMortality(v, s);
    }
    
    double computeProbAcceptance(double oldFitness, double newFitness, double popSize) {
        double selAdv = (newFitness - oldFitness)/(1.0E-10 + oldFitness); 
        if ((Math.abs(selAdv) * popSize) < 0.01) {
            return 1.0;
        }
        if (selAdv > 10.0) {
            return 2.0 * popSize;
        } 
        if (popSize * selAdv > 10.0) {
            return  2.0 * popSize * (1.0 - Math.exp(-2.0 * selAdv));
        }
        if (selAdv * popSize < -10.0) {
            return 0.0;
        }
        return 2.0 * popSize * (1.0 - Math.exp(-2.0 * selAdv)) / (1.0 - Math.exp(-4.0 * selAdv * popSize));
    }
    
    double computeMortality(double v, double s) {
        double sMortality = (nS * onePlusEpsilon * s / (onePlusEpsilon - s)); 
        double vMortality =  (1 - effectiveness*(s0 + s)/(s0 + 1)) * (nV * onePlusEpsilon * v / (onePlusEpsilon-v));
        double mortality = d0 + sMortality + vMortality;
        return mortality;
    }
    
    double computeTransmission(double v, double s) {
        double transmission = (1 - effectiveness*(s0 + s)/(s0 + 1)) * (v0 + v);
        return transmission;
    }
    
    void findMaxFit() {
        for (int i = 0; i < 10001; i++) {
            double v = i * 0.0001;
            fitPathMax = Math.max(fitPathMax, computePathFitness(v, 0.0));
            double s = i * 0.0001;
            fitHostMax = Math.max(fitHostMax, computeHostFitness(0.0, s));
        }
    }
    
    /**
     * Makes series of step sizes that are equally likely
     */
    void makeSteps() {
        double[] divides = new double[nSteps+1];
        for (int iDiv = 1; iDiv < nSteps; iDiv++) {
            divides[iDiv] = sqrt2*Erf.erfInv((2.0*iDiv)/nSteps-1.0);
        }
        divides[0] = divides[1]*10.0;
        divides[nSteps] = divides[nSteps-1]*10.0;
        for (int i = 0; i < nSteps; i++) {
            possibleSteps[i] = (Math.exp(-divides[i]*divides[i]/2.0)
                    - Math.exp(-divides[i+1]*divides[i+1]/2.0))/(sqrt2pi/nSteps);
        }
    }
    
    
    HashMap<String, double[]> findPoints(double inputBV, double inputMV, double inputBS, double inputMS) {

        HashMap<String, double[]> vsPairsList = new HashMap<>();
        
        double[] vs = new double[2];
        vs[1] = (inputBS + inputMS * inputBV)/(1.0 - inputMS * inputMV);
        vs[0] = inputBV + inputMV * vs[1];
        if (Math.abs(inputMS*inputMV) < 1.0 && vs[0] > 0.0 && vs[0] < 1.0 && vs[1] > 0.0 && vs[1] < 1.0) {
            String tag = Arrays.toString(vs);
            vsPairsList.put(tag, vs);
            return vsPairsList;
        }

        double s0 = Math.max(0.0, Math.min(1.0,inputBS));
        double v_s0 = Math.max(0.0, Math.min(1.0,inputBV + inputMV * s0));
        if (v_s0 < 0.00000001) {
            vs = new double[2];
            vs[0] = v_s0;
            vs[1] = s0;
            String tag = Arrays.toString(vs);
            if (!vsPairsList.containsKey(tag)) {
                vsPairsList.put(tag, vs);
            }
        }
        
        double s1 = Math.max(0.0, Math.min(1.0,inputBS+inputMS));
        double v_s1 = Math.max(0.0, Math.min(1.0,inputBV + inputMV * s1));
        if (v_s1 > 0.99999999) {
            vs = new double[2];
            vs[0] = v_s1;
            vs[1] = s1;
            String tag = Arrays.toString(vs);
            if (!vsPairsList.containsKey(tag)) {
                vsPairsList.put(tag, vs);
            } 
        }  
        
        double v0 = Math.max(0.0, Math.min(1.0,inputBV));
        double s_v0 = Math.max(0.0, Math.min(1.0,inputBS + inputMS * v0));
        if (s_v0 < 0.00000001) {
            vs = new double[2];
            vs[0] = v0;
            vs[1] = s_v0;
            String tag = Arrays.toString(vs);
            if (!vsPairsList.containsKey(tag)) {
                vsPairsList.put(tag, vs);
            }
        }
        
        double v1 = Math.max(0.0, Math.min(1.0,inputBV+inputMV));
        double s_v1 = Math.max(0.0, Math.min(1.0,inputBS + inputMS * v1));
        if (s_v1 > 0.99999999) {
            vs = new double[2];
            vs[0] = v1;
            vs[1] = s_v1;
            String tag = Arrays.toString(vs);
            if (!vsPairsList.containsKey(tag)) {
                vsPairsList.put(tag, vs);
            }
        }       
        return vsPairsList;
    }
    
    
}
