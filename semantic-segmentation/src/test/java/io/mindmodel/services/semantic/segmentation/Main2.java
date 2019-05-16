package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.attic.GraphicsUtils;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Concat;
import org.tensorflow.op.core.ExpandDims;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Range;
import org.tensorflow.op.core.ReduceMax;
import org.tensorflow.op.core.Squeeze;
import org.tensorflow.op.core.Tile;
import org.tensorflow.op.dtypes.Cast;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.image.EncodeJpeg;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.math.Sub;
import org.tensorflow.types.UInt8;

/**
 * @author Christian Tzolov
 */
public class Main2 {



	public static void compute1() throws IOException {

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi.jpg");
//		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/giant_panda_in_beijing_zoo_1.jpg");

		Graph g = new Graph();

		Ops tf = Ops.create(g);

		Placeholder<String> input = tf.withName("input_image").placeholder(String.class);
		DecodeJpeg decodedImage = tf.withName("decoded").image.decodeJpeg(input, DecodeJpeg.channels(3L));

		EncodeJpeg encode = tf.withName("encode").image.encodeJpeg(decodedImage);

		Session s = new Session(g);

		try (Tensor inputTensor = Tensor.create(inputImage)) {
			List<Tensor<?>> result = s.runner()
					.feed("input_image", inputTensor)
					.fetch("decoded")
					.fetch("encode")
					.run();

			System.out.println(result);
			byte[] bb = inputTensor.bytesValue();
			System.out.println(inputTensor.numBytes());
			System.out.println(bb);
		}
	}

	/**
	 * https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/image_ops_impl.py#L1536
	 * @param tf
	 * @param images
	 * @return
	 */
	public static <T> Operand<T> grayscaleToRgb(Ops tf, Operand<T> images) {
		ExpandDims<Integer> rank_1 = tf.expandDims(
				tf.math.sub(tf.rank(images), tf.constant(1)),
				tf.constant(0));
		// Create once 1D vector of the shape defined by the rank_1.
		// E.g. for rank [2] will produce matrix [1, 1]. For [3] rank will produce a cube [1, 1, 1]
		Add<Integer> ones = tf.math.add(tf.zeros(rank_1, Integer.class), tf.constant(1));
		// Convert scalar 3 into 1D array [3]
		ExpandDims<Integer> channelsAs1D = tf.expandDims(tf.constant(3), tf.constant(0));
		Concat<Integer> shapeList = tf.concat(Arrays.asList(ones, channelsAs1D), tf.constant(0));
		Tile<T> tile = tf.withName("grayscaleToRgb").tile(images, shapeList);
		return tile;
	}

	public static Operand<Float> normalizeMask(Ops tf, Operand<Float> mask, float newValue) {
		// generate array representing the axis indexes.
		// For example of tensor of rank K the axisRange is {0, 1, 2 ...K}
		Range<Integer> axisRange = tf.range(tf.constant(0),  // from
				tf.dtypes.cast(tf.rank(mask), Integer.class), // to
				tf.constant(1)); // step

		ReduceMax<Float> max = tf.reduceMax(mask, axisRange);
		//Mul<Float> input2Float1 = tf.math.mul(tf.math.div(input2Float, max), tf.constant(1f));
		Mul<Float> normalizedMask = tf.math.mul(tf.math.div(mask, max), tf.constant(newValue));

		return normalizedMask;
	}

	/**
	 * https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
	 * @param tf
	 * @param srcRgb
	 * @param dstRgb
	 * @param srcAlpha
	 * @return
	 */
	public static Operand<Float> alphaBlending(Ops tf, Operand<Float> srcRgb, Operand<Float> dstRgb, float srcAlpha) {


		Sub<Float> alpha = tf.math.sub(tf.onesLike(srcRgb), tf.constant(srcAlpha));
		Mul<Float> src = tf.math.mul(srcRgb, alpha);
		Mul<Float> dst = tf.math.mul(dstRgb, tf.math.sub(tf.constant(1.0f), alpha));
		Add<Float> out = tf.math.add(dst, src);

		//Mul<Float> out = tf.math.mul(srcRgbNormalized, dstRgb);

		//Squeeze<Float> squeeze = tf.withName("squeeze").squeeze(out, Squeeze.axis(Arrays.asList(0L)));

		return out;
	}

	public static Operand<Float> alphaBlending(Ops tf, Operand<Float> srcRgb, Operand<Float> dstRgb, Operand<Float> srcAlpha) {


		Sub<Float> alpha = tf.math.sub(tf.onesLike(srcRgb), srcAlpha);
		Mul<Float> src = tf.math.mul(srcRgb, alpha);
		Mul<Float> dst = tf.math.mul(dstRgb, tf.math.sub(tf.constant(1.0f), alpha));
		Add<Float> out = tf.math.add(dst, src);

		//Mul<Float> out = tf.math.mul(srcRgbNormalized, dstRgb);

		//Squeeze<Float> squeeze = tf.withName("squeeze").squeeze(out, Squeeze.axis(Arrays.asList(0L)));

		return out;
	}




	public static void compute2(Tensor<?> tensor) throws IOException {

		Graph g = new Graph();

		Ops tf = Ops.create(g);  // Normalize image eagerly, using default session

		Placeholder<Long> input = tf.withName("input_tensor").placeholder(Long.class);
		ExpandDims<Long> expanded = tf.withName("expanded").expandDims(input, tf.constant(3));

		Squeeze<Long> squeeze = tf.withName("squeeze").squeeze(expanded, Squeeze.axis(Arrays.asList(0L)));


		ExpandDims<Integer> rank_1 = tf.withName("rank_1").expandDims(tf.math.sub(tf.rank(squeeze), tf.constant(1)), tf.constant(0));
		Add<Integer> rank_11 = tf.withName("rank_11").math.add(tf.zeros(rank_1, Integer.class), tf.constant(1));
		//ExpandDims<Integer> rank_12 = tf.withName("rank_12").expandDims(tf.constant(3), tf.constant(0));
		ExpandDims<Integer> rank_12 = tf.withName("rank_12").expandDims(tf.constant(2), tf.constant(0));
		Concat<Integer> shapeList = tf.withName("shapeList").concat(Arrays.asList(rank_11, rank_12), tf.constant(0));
		Tile<Long> tile = tf.withName("tile").tile(squeeze, shapeList);

		//Mul<Long> b = tf.math.mul(tf.math.div(squeeze, tf.constant(12L)), tf.constant(127L));
		Mul<Long> b = tf.math.mul(tf.math.sub(tf.math.div(tile, tf.constant(12L)), tf.constant(1L)), tf.constant(-127L));
		//Mul<Long> b = tf.math.mul(tf.math.div(tile, tf.constant(12L)), tf.constant(127L));


		ReduceMax<Long> max = tf.reduceMax(squeeze, tf.constant(new long[] { 0, 1, 2 }));
		Mul<Long> squeeze2 = tf.math.mul(tf.math.div(squeeze, max), tf.constant(255L));
		Mul<Long> ones = tf.math.mul(tf.onesLike(squeeze2), tf.constant(255L));
		Concat<Long> concat = tf.withName("concat").concat(Arrays.asList(squeeze2, ones), tf.constant(2L));

		Cast<UInt8> cast = tf.withName("cast1").dtypes.cast(b, UInt8.class);
		//Cast<UInt8> cast = tf.withName("cast1").dtypes.cast(concat, UInt8.class);

		//Operand<String> jpeg = tf.withName("jpeg").encodeJpeg(cast);
		Operand<String> png = tf.withName("png").image.encodePng(cast);


		Session s = new Session(g);

		List<Tensor<?>> result = s.runner()
				.feed("input_tensor", tensor)
				.fetch("expanded")
				.fetch("squeeze")
				.fetch("cast1")
				.fetch("png")
				.fetch("rank_1")
				.fetch("shapeList")
				.fetch("rank_11")
				.fetch("rank_12")
				.fetch("tile")
				.fetch("concat")
				.run();

		long[][][] boza = new long[(int) result.get(8).shape()[0]][(int) result.get(8).shape()[1]][(int) result.get(8).shape()[2]];
		result.get(9).copyTo(boza);

		for (int x = 0; x < boza.length; x++) {
			for (int y = 0; y < boza[0].length; y++) {
				if (boza[x][y][0] > 0) {
					System.out.println(x + ":" + y + " = " + boza[x][y][0] + " - " + boza[x][y][1]);
				}
			}
		}

		//int[] shapeList1 = new int[1];
		//result.get(5).copyTo(shapeList1);


		//System.out.println(Arrays.toString(boza));
		byte[] mask = result.get(3).bytesValue();
		System.out.println(result);
		int[] r = new int[1];
		result.get(4).copyTo(r);

		int[] sh = new int[2];
		result.get(6).copyTo(sh);

		int[] sh2 = new int[1];
		result.get(7).copyTo(sh2);

		int[] shl = new int[3];
		result.get(5).copyTo(shl);

		ByteArrayInputStream bis = new ByteArrayInputStream(mask);
		BufferedImage bImage2 = ImageIO.read(bis);
		ImageIO.write(bImage2, "png", new File("./semantic-segmentation/target/outputMaskXX.png"));
	}

	public static void main(String[] args) throws IOException {
		//compute1();
		//Tensor<?> t = boza();
		//compute2(t);

		//List<? extends Tensor<?>> images = compute3();
		//AutoCloseables.all(images);
	}
}
