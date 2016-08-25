package noam.bar.myapplication;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        int flag = 0;
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            trainMLP();
        } catch (Exception e) {
            Log.w("crap", e.getMessage());
            flag = 1;
        }
        if (flag == 1)
            setContentView(R.layout.bad_end);
        else
            setContentView(R.layout.good_end);

    }


    public void trainMLP() throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 10000;
        int batchSize = 500;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = iterations / 5;
        int splitTrainNum = (int) (batchSize * .8);
    }

  /*  public void trainMLP() throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int batchSize = 128;
        int rngSeed = 123;
        int numEpochs = 15;



        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        for( int i=0; i<1; i++ ){
            model.fit(mnistTrain);
        }

    }*/
}
