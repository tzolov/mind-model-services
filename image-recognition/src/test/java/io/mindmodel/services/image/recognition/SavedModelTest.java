package io.mindmodel.services.image.recognition;

import java.util.Iterator;
import java.util.Map;

import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;

/**
 * @author Christian Tzolov
 */
public class SavedModelTest {

	/**
	 * https://medium.com/@jsflo.dev/saving-and-loading-a-tensorflow-model-using-the-savedmodel-api-17645576527
	 *
	 * https://www.tensorflow.org/alpha/guide/saved_model
	 *
	 * @param args
	 */
	public static void main(String[] args) throws InvalidProtocolBufferException {
		SavedModelBundle savedModelBundle =
				SavedModelBundle.load("/Users/ctzolov/Downloads/ssd_mobilenet_v1_coco_2017_11_17/saved_model", "serve");
		//SavedModelBundle savedModelBundle =
		//		SavedModelBundle.load("/Users/ctzolov/Downloads/mnasnet-a1/saved_model", "serve");

		MetaGraphDef meta = MetaGraphDef.parseFrom(savedModelBundle.metaGraphDef());

		Map<String, SignatureDef> signatures = meta.getSignatureDefMap();

		System.out.println(signatures);

		savedModelBundle.session();

		//Iterator<Operation> itr = savedModelBundle.graph().operations();
		//
		//while (itr.hasNext()) {
		//	System.out.println("Operation: " + itr.next());
		//}
	}
}
