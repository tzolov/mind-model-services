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
package io.mindmodel.services.pose.estimation.benchmark;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import io.mindmodel.services.common.GraphicsUtils;
import io.mindmodel.services.common.TensorFlowService;
import io.mindmodel.services.pose.estimation.PoseEstimationTensorflowInputConverter;
import io.mindmodel.services.pose.estimation.PoseEstimationTensorflowOutputConverter;
import io.mindmodel.services.pose.estimation.domain.Body;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.runner.RunnerException;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;

/**
 * @author Christian Tzolov
 */
public class PoseEstimationServiceBenchmark {

	@State(Scope.Benchmark)
	public static class ExecutionPlan {
		public Function<byte[][], List<List<Body>>> poseEstimationService1;
		private Function<byte[][], List<List<Body>>> poseEstimationService2;
		public byte[][] images;

		@Setup(Level.Trial)
		public void setUp() throws IOException {
			ResourceLoader resourceLoader = new DefaultResourceLoader();
			Resource modelResource1 = resourceLoader.getResource("https://dl.bintray.com/big-data/generic/2018-30-05-mobilenet_thin_graph_opt.pb");
			Resource modelResource2 = resourceLoader.getResource("https://dl.bintray.com/big-data/generic/2018-05-14-cmu-graph_opt.pb");
			List<String> fetchNames = Arrays.asList("Openpose/concat_stage7");

			PoseEstimationTensorflowInputConverter inputConverter = new PoseEstimationTensorflowInputConverter();
			PoseEstimationTensorflowOutputConverter outputConverter = new PoseEstimationTensorflowOutputConverter(fetchNames);

			poseEstimationService1 = inputConverter.andThen(new TensorFlowService(modelResource1, fetchNames)).andThen(outputConverter);
			poseEstimationService2 = inputConverter.andThen(new TensorFlowService(modelResource2, fetchNames)).andThen(outputConverter);
			images = new byte[][] { GraphicsUtils.toImageToBytes("classpath:/images/VikiMaxiAdi2.jpg") };
		}
	}

	@Fork(value = 1, warmups = 1)
	@Benchmark
	@BenchmarkMode(Mode.AverageTime)
	@Threads(value = 1)
	public void poseEstimationThinGraph(ExecutionPlan plan) {
		List<List<Body>> result = plan.poseEstimationService1.apply(plan.images);
	}

	@Fork(value = 1, warmups = 1)
	@Benchmark
	@BenchmarkMode(Mode.AverageTime)
	@Threads(value = 1)
	public void poseEstimationCmuGraph(ExecutionPlan plan) {
		List<List<Body>> result = plan.poseEstimationService2.apply(plan.images);
	}

	public static void main(String[] args) throws IOException, RunnerException {
		org.openjdk.jmh.Main.main(args);
	}
}
