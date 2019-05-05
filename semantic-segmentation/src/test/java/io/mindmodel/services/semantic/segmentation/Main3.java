package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.attic.GraphicsUtils;
import io.mindmodel.services.common.attic.TensorFlowService;
import io.mindmodel.services.semantic.segmentation.attic.SemanticSegmentationUtils;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Gather;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Squeeze;
import org.tensorflow.op.core.Where3;
import org.tensorflow.op.math.Mod;
import org.tensorflow.op.math.Sub;
import org.tensorflow.types.UInt8;

import org.springframework.core.io.DefaultResourceLoader;

/**
 * @author Christian Tzolov
 */
public class Main3 {

	public static Tensor<?> detectMask() throws IOException {
		SegmentationInputConverter inputConverter = new SegmentationInputConverter();
		Map<String, Tensor<?>> inTensors = inputConverter.apply(GraphicsUtils.loadAsByteArray("classpath:/images/interior.jpg"));
		TensorFlowService service = new TensorFlowService(new DefaultResourceLoader().getResource("http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz#frozen_inference_graph.pb")
				, Arrays.asList(SemanticSegmentationUtils.OUTPUT_TENSOR_NAME), true);
		Map<String, Tensor<?>> outTensors = service.apply(inTensors);
		System.out.println(outTensors);
		return outTensors.get(SemanticSegmentationUtils.OUTPUT_TENSOR_NAME);
	}

	public static void main(String[] args) throws IOException {


		//Tensor<?> colorTableTensor = Tensor.create(mapillaryColormap);
		Tensor<?> colorTableTensor = Tensor.create(SegmentationColorMap.ADE20K_COLORMAP);

		System.out.println(Arrays.toString(colorTableTensor.shape()));

		//int[][] l = new int[3][20];
		//for (int x = 0; x < 3; x++) {
		//	for (int y = 0; y < 20; y++) {
		//		l[x][y] = 0;
		//	}
		//}
		//
		//l[1][1] = 15;
		//l[2][1] = 100;

//		Tensor<?> maskTensor = Tensor.create(l);
//		System.out.println(Arrays.toString(maskTensor.shape()));

		Graph g = new Graph();

		Ops tf = Ops.create(g);
		Placeholder<Integer> colorTable = tf.withName("table").placeholder(Integer.class);

		Placeholder<Long> mask = tf.withName("maskImage").placeholder(Long.class);

		Squeeze<Long> mask2 = tf.squeeze(mask, Squeeze.axis(Arrays.asList(0L)));

		Operand cond = tf.math.lessEqual(mask2, tf.math.add(tf.onesLike(mask2), tf.constant(12L)));
		Where3 whereMask = tf.withName("cond").where3(cond, tf.onesLike(mask2), tf.zerosLike(mask2));
		//tf.withName("where").gather(mask2, whereInx, tf.constant(1));

		Sub<Long> colorTableShape = tf.math.sub(tf.shape(colorTable, Long.class), tf.constant(1L));
		Gather<Long> colorTableSize = tf.withName("gather").gather(colorTableShape, tf.constant(new int[] { 0 }), tf.constant(0));


		//Mod<Long> mask3 = tf.math.mod(mask2, colorTableSize);
		Mod<Long> mask3 = tf.math.mod(whereMask, colorTableSize);

		//tf.withName("output").gatherNd(table, tf.constant(new int[]{0, 1}));
		//tf.withName("output").gather(table, tf.constant(new int[]{0, 1}), tf.constant(0));
		Gather<Integer> rgb = tf.withName("rgb").gather(colorTable, mask3, tf.constant(0));

		Operand<String> png = tf.withName("png").image.encodePng(tf.dtypes.cast(rgb, UInt8.class));

		Session s = new Session(g);

		List<Tensor<?>> result = s.runner()
				.feed("table", colorTableTensor)
				.feed("maskImage", detectMask())
				.fetch("rgb")
				.fetch("png")
				.fetch("gather")
				.fetch("cond")
				.run();

		System.out.println(result);
		Tensor<?> result0 = result.get(0);
		int[][][] r = new int[(int) result0.shape()[0]][(int) result0.shape()[1]][(int) result0.shape()[2]];
		result0.copyTo(r);

		byte[] maskPng = result.get(1).bytesValue();

		ByteArrayInputStream bis = new ByteArrayInputStream(maskPng);
		BufferedImage bImage2 = ImageIO.read(bis);
		ImageIO.write(bImage2, "png", new File("./semantic-segmentation/target/maskPng.png"));

		System.out.println(Arrays.toString(result.get(2).shape()));
		long[] sss = new long[1];
		result.get(2).copyTo(sss);

		System.out.println(sss[0]);

		System.out.println(Arrays.toString(result.get(3).shape()));
		//System.out.println(Arrays.toString(result.get(4).shape()));

	}
}
