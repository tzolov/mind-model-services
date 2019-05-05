package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.attic.GraphicsUtils;
import io.mindmodel.services.semantic.segmentation.attic.SemanticSegmentationService;
import org.apache.commons.io.IOUtils;

/**
 * @author Christian Tzolov
 */
public class SemanticSegmentationExample2 {


	public static void main(String[] args) throws IOException {
		String MODEL_URI =
				"http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz#frozen_inference_graph.pb";
		SemanticSegmentationService segmentationService = new SemanticSegmentationService(MODEL_URI, true);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi.jpg");

		// Read get the segmentation maskImage as separate image
		byte[] imageMask = segmentationService.masksAsImage(inputImage);
		writeImage(imageMask, "png", "./semantic-segmentation/target/VikiMaxiAdi_masks.png");

		// Blend the segmentation maskImage on top of the original image
		byte[] augmentedImage = segmentationService.augment(inputImage);
		IOUtils.write(augmentedImage,
				new FileOutputStream("./semantic-segmentation/target/VikiMaxiAdi_augmented.jpg"));
	}

	private static void writeImage(byte[] image, String imageFormat, String outputPath) throws IOException {
		BufferedImage i1 = ImageIO.read(new ByteArrayInputStream(image));
		ImageIO.write(i1, imageFormat, new FileOutputStream(outputPath));
	}

}
