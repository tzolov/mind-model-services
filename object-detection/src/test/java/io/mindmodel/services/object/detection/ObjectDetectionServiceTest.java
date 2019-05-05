/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.mindmodel.services.object.detection;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.stream.Collectors;

import io.mindmodel.services.common.attic.JsonMapperFunction;
import io.mindmodel.services.object.detection.domain.ObjectDetection;
import org.apache.commons.io.IOUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.ResourceLoader;
import org.springframework.util.StreamUtils;

/**
 * @author Christian Tzolov
 */
public class ObjectDetectionServiceTest {

	ResourceLoader resourceLoader = new DefaultResourceLoader();

	String labelResource = "http://dl.bintray.com/big-data/generic/mscoco_label_map.pbtxt";
	String modelResource = "http://dl.bintray.com/big-data/generic/ssdlite_mobilenet_v2_coco_2018_05_09_frozen_inference_graph.pb";

	@Test
	public void testObjectDetection() throws IOException {
		ObjectDetectionService objectDetectionService =
				new ObjectDetectionService(modelResource, labelResource, 0.4f, false, true);
		try (InputStream is = new ClassPathResource("/images/object-detection.jpg").getInputStream()) {

			byte[] image = StreamUtils.copyToByteArray(is);

			List<ObjectDetection> detectedObjects = objectDetectionService.detect(image);

			Assert.assertNotNull(detectedObjects);
			Assert.assertEquals(8, detectedObjects.size());
		}
	}

	@Ignore
	@Test
	public void testObjectDetection2() throws IOException {
		ObjectDetectionService objectDetectionService =
				new ObjectDetectionService(modelResource, labelResource, 0.4f, false, true);
		try (InputStream is = new ClassPathResource("/images/object-detection.jpg").getInputStream()) {

			byte[] image = StreamUtils.copyToByteArray(is);

			List<ObjectDetection> detectedObjects = objectDetectionService.detect(image);

			byte[] annotatedImage = new ObjectDetectionImageAugmenter().apply(image, detectedObjects);

			IOUtils.write(annotatedImage, new FileOutputStream("./target/out2.jpg"));

			Assert.assertNotNull(detectedObjects);
			Assert.assertEquals(8, detectedObjects.size());
		}
	}


	@Ignore
	@Test
	public void testObjectDetection3() throws IOException {
		//
		String modelResource = "file:///Users/ctzolov/Downloads/faster_rcnn_resnet101_fgvc_2018_07_19/frozen_inference_graph.pb";
		//String modelResource = "http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_fgvc_2018_07_19.tar.gz#frozen_inference_graph.pb";
		String labelResource = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/fgvc_2854_classes_label_map.pbtxt";
		ObjectDetectionService objectDetectionService =
				new ObjectDetectionService(modelResource, labelResource, 0.1f, false, false);
		try (InputStream is = new ClassPathResource("/images/north-american-wildlife-7.gif").getInputStream()) {

			byte[] image = StreamUtils.copyToByteArray(is);

			List<ObjectDetection> detectedObjects = objectDetectionService.detect(image);

			byte[] annotatedImage = new ObjectDetectionImageAugmenter().apply(image, detectedObjects);

			IOUtils.write(annotatedImage, new FileOutputStream("./target/out2.jpg"));

			Assert.assertNotNull(detectedObjects);
		}
	}

	@Test
	public void testObjectDetection4() throws IOException {
		String labelResource = "http://dl.bintray.com/big-data/generic/mscoco_label_map.pbtxt";
		String modelResource = "http://dl.bintray.com/big-data/generic/ssdlite_mobilenet_v2_coco_2018_05_09_frozen_inference_graph.pb";

		ObjectDetectionService objectDetectionService = new ObjectDetectionService(modelResource, labelResource, 0.4f, false, false);

		byte[][] images = new byte[3][];
		images[0] = imageToBytes("classpath:/images/object-detection.jpg");
		images[1] = imageToBytes("classpath:/images/object-detection.jpg");
		images[2] = imageToBytes("classpath:/images/object-detection.jpg");

		List<List<ObjectDetection>> out = objectDetectionService.detect(images);

		Assert.assertEquals(3, out.size());

		List<String> jsonList = out.stream().map(new JsonMapperFunction()).collect(Collectors.toList());
		//System.out.println(jsonList.get(0));

		Assert.assertEquals("[{\"name\":\"kite\",\"confidence\":0.8673682,\"x1\":0.44308642,\"y1\":0.0814952,\"x2\":0.5014239,\"y2\":0.169772,\"cid\":38}," +
				"{\"name\":\"kite\",\"confidence\":0.80015683,\"x1\":0.34496674,\"y1\":0.37845963,\"x2\":0.3610665,\"y2\":0.4024166,\"cid\":38}," +
				"{\"name\":\"person\",\"confidence\":0.787643,\"x1\":0.3913834,\"y1\":0.5630007,\"x2\":0.4083495,\"y2\":0.59502745,\"cid\":1}," +
				"{\"name\":\"person\",\"confidence\":0.7242934,\"x1\":0.08181207,\"y1\":0.68023187,\"x2\":0.124822214,\"y2\":0.83192676,\"cid\":1}," +
				"{\"name\":\"person\",\"confidence\":0.62905985,\"x1\":0.059143387,\"y1\":0.57875264,\"x2\":0.07551455,\"y2\":0.61880136,\"cid\":1}," +
				"{\"name\":\"person\",\"confidence\":0.6122513,\"x1\":0.0257214,\"y1\":0.5782344,\"x2\":0.04140708,\"y2\":0.6188131,\"cid\":1}," +
				"{\"name\":\"kite\",\"confidence\":0.60772794,\"x1\":0.20563951,\"y1\":0.27496633,\"x2\":0.22761866,\"y2\":0.3100944,\"cid\":38}," +
				"{\"name\":\"person\",\"confidence\":0.5325221,\"x1\":0.15765251,\"y1\":0.76527464,\"x2\":0.20344453,\"y2\":0.9485351,\"cid\":1}]", jsonList.get(0));

		Assert.assertEquals(jsonList.get(0), jsonList.get(1));
		Assert.assertEquals(jsonList.get(0), jsonList.get(2));
	}

	private static byte[] imageToBytes(String imageUri) throws IOException {
		try (InputStream is = new DefaultResourceLoader().getResource(imageUri).getInputStream()) {
			return StreamUtils.copyToByteArray(is);
		}
	}

}
