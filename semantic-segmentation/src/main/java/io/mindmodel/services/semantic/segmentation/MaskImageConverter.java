package io.mindmodel.services.semantic.segmentation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import io.mindmodel.services.common.GraphRunner;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Gather;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Squeeze;
import org.tensorflow.op.core.ZerosLike;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Equal;
import org.tensorflow.types.UInt8;

/**
 * Blends the transparent maskImage on top of the input image
 *
 * @author Christian Tzolov
 */
public class MaskImageConverter extends GraphRunner {


	private final Tensor<Integer> colorTableTensor;
	private long[] labelFilter;

	public MaskImageConverter(int[][] colorMap) {
		this(colorMap, null);
	}

	public MaskImageConverter(int[][] colorMap, long labelFilter[]) {
		super(Arrays.asList("color_map", "mask_pixels"), Arrays.asList("mask_png", "mask_rgb"), false);
		this.colorTableTensor = Tensor.create(colorMap).expect(Integer.class);
		this.labelFilter = labelFilter;
	}

	@Override
	public void doClose() {
		this.colorTableTensor.close();
	}

	@Override
	protected void doGraphDefinition(Ops tf) {

		Placeholder<Integer> colorTable = tf.withName("color_map").placeholder(Integer.class);

		Placeholder<Long> batchedMask = tf.withName("mask_pixels").placeholder(Long.class);
		// Remove batch dimension
		Squeeze<Long> mask = tf.squeeze(batchedMask, Squeeze.axis(Arrays.asList(0L)));

		Operand<Long> filteredMask = labelFilter(tf, mask, this.labelFilter);

		// The mask can contain label values larger than the list of colors provided in the color map.
		// To avoid out-of-index errors we will "normalize" the label values in the mask to MOD max-color-table-value.
		Operand<Long> mask3 = NativeImageUtils.normalizeMaskLabels(tf, colorTable, filteredMask);

		Gather<Integer> maskRgb = tf.withName("mask_rgb").gather(colorTable, mask3, tf.constant(0));

		Operand<String> png = tf.withName("mask_png").image.encodePng(tf.dtypes.cast(maskRgb, UInt8.class));
	}

	private Operand<Long> labelFilter(Ops tf, Operand<Long> mask, long[] labels) {

		if (labels == null || labels.length == 0) {
			return mask;
		}

		ZerosLike<Long> zeroMask = tf.zerosLike(mask);
		Operand<Long> result = zeroMask;
		for (long label : labels) {
			Add<Long> labelMask = tf.math.add(tf.zerosLike(mask), tf.constant(label));
			Equal condition = tf.math.equal(mask, labelMask);
			result = tf.math.add(result, tf.where3(condition, labelMask, zeroMask));
		}
		return result;
	}

	@Override
	public Map<String, Tensor<?>> apply(Map<String, Tensor<?>> feeds) {

		Map<String, Tensor<?>> all = new HashMap<>();
		all.putAll(feeds);
		all.put("color_map", colorTableTensor);

		return super.apply(all);
	}
}
