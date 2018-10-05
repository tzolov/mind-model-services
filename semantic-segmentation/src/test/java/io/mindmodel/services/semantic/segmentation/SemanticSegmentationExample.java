package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.GraphicsUtils;
import org.apache.commons.io.IOUtils;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;

/**
 * @author Christian Tzolov
 */
public class SemanticSegmentationExample {


	public static void main(String[] args) throws IOException {

		Resource PASCAL_VOC_2012_MODEL = new DefaultResourceLoader().getResource(
				"http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz#frozen_inference_graph.pb");
		SemanticSegmentationService segmentationService = new SemanticSegmentationService(PASCAL_VOC_2012_MODEL, true);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi.jpg");

		// Read get the segmentation mask as separate image
		byte[] imageMask = segmentationService.masksAsImage(inputImage);
		writeImage(imageMask, "png", "./semantic-segmentation/target/VikiMaxiAdi_masks.png");

		// Blend the segmentation mask on top of the original image
		byte[] augmentedImage = segmentationService.augment(inputImage);
		IOUtils.write(augmentedImage,
				new FileOutputStream("./semantic-segmentation/target/VikiMaxiAdi_augmented.jpg"));
	}

	private static void writeImage(byte[] image, String imageFormat, String outputPath) throws IOException {
		BufferedImage i1 = ImageIO.read(new ByteArrayInputStream(image));
		ImageIO.write(i1, imageFormat, new FileOutputStream(outputPath));
	}

}
