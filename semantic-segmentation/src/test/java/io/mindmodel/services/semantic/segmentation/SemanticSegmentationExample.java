package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.attic.GraphicsUtils;
import org.apache.commons.io.IOUtils;

/**
 * @author Christian Tzolov
 */
public class SemanticSegmentationExample {


	public static void main(String[] args) throws IOException {
		String PASCAL_VOC_2012_MODEL =
				"http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz#frozen_inference_graph.pb";
		SemanticSegmentation segmentation = new SemanticSegmentation(PASCAL_VOC_2012_MODEL,
				SegmentationColorMap.ADE20K_COLORMAP, null, 0.45f);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi.jpg");

		// Read get the segmentation maskImage as separate image
		byte[] imageMask = segmentation.maskImage(inputImage);
		writeImage(imageMask, "png", "./semantic-segmentation/target/VikiMaxiAdi_masks.png");

		// Blend the segmentation maskImage on top of the original image
		byte[] augmentedImage = segmentation.blendMask(inputImage);
		IOUtils.write(augmentedImage,
				new FileOutputStream("./semantic-segmentation/target/VikiMaxiAdi_augmented.jpg"));
	}

	private static void writeImage(byte[] image, String imageFormat, String outputPath) throws IOException {
		BufferedImage i1 = ImageIO.read(new ByteArrayInputStream(image));
		ImageIO.write(i1, imageFormat, new FileOutputStream(outputPath));
	}

}
