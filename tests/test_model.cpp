#include <gtest/gtest.h>

#include "core/common.h"
#include "cuda/cuda_utils.h"
#include "model/model.h"

namespace fe = cudnn_frontend;

class ModelTest : public ::testing::Test
{
   protected:
    cudnnHandle_t cudnn_handle_;
    std::shared_ptr<fe::graph::Graph> graph_for_test_;  // 테스트용 그래프 객체

    // 테스트용 float 배열 비교 헬퍼
    static bool AreFloatsClose(float actual, float expected, float tolerance = 1e-5f)
    {
        return std::abs(actual - expected) < tolerance;
    }

    static void PrintTensorData(const std::string& name, const ushionn::Tensor& tensor, size_t max_elements = 10)
    {
        std::cout << "Tensor: " << name << " (UID: " << tensor.get_uid() << ", Shape: [";
        for (size_t i = 0; i < tensor.get_shape().size(); ++i)
        {
            std::cout << tensor.get_shape()[i] << (i == tensor.get_shape().size() - 1 ? "" : ", ");
        }
        std::cout << "], Location: " << static_cast<int>(tensor.get_data_location()) << ")" << std::endl;

        if (!tensor.is_on_host() && !tensor.is_on_device())
        {
            std::cout << "  Data not available for printing." << std::endl;
            return;
        }

        ushionn::Tensor host_tensor = tensor;  // 복사본 생성
        if (host_tensor.get_data_location() == ushionn::Tensor::DataLocation::DEVICE)
        {
            std::cout << "  Transferring to host for printing..." << std::endl;
            host_tensor.to_host();  // CPU로 가져오기
        }

        if (host_tensor.get_data_location() == ushionn::Tensor::DataLocation::HOST)
        {
            const float* data_ptr = static_cast<const float*>(host_tensor.get_host_ptr());
            if (data_ptr)
            {
                std::cout << "  Data: [";
                size_t count = 0;
                for (size_t i = 0; i < host_tensor.get_num_elements() && count < max_elements; ++i, ++count)
                {
                    std::cout << data_ptr[i]
                              << (i == host_tensor.get_num_elements() - 1 || count == max_elements - 1 ? "" : ", ");
                }
                if (host_tensor.get_num_elements() > max_elements)
                {
                    std::cout << ", ...";
                }
                std::cout << "]" << std::endl;
            }
            else
            {
                std::cout << "  Host data pointer is null." << std::endl;
            }
        }
        else
        {
            std::cout << "  Could not bring data to host for printing." << std::endl;
        }
    }

    void SetUp() override
    {
        CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
        // graph_for_test_는 모델이 내부적으로 생성/관리하므로, 여기서는 직접 사용하지 않을 수 있음.
        // 모델 생성 시 cudnn_handle_을 전달.
        ushionn::cuda::utils::printGpuMemoryUsage("Before predict test case");
    }

    void TearDown() override
    {
        if (cudnn_handle_)
        {
            CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
            cudnn_handle_ = nullptr;
        }
        ushionn::cuda::utils::printGpuMemoryUsage("After predict test case");
    }
};

TEST_F(ModelTest, SingleDenseLayerNoActivation)
{
    // 1. 모델 구성
    ushionn::model::Sequential model(cudnn_handle_);  // 모델에 cudnnHandle 전달
    // ---> [내가 채울 부분] 아래 DenseLayer 생성자 인자를 실제 구현에 맞게 수정.
    //      (input_size, output_size, name, has_bias=false 등)
    //      예: auto dense_layer = std::make_unique<ushionn::DenseLayer>(2, 3, "dense1", false); // 입력 2, 출력 3, bias
    //      없음
    int64_t input_features = 2;
    int64_t output_features = 3;
    auto dense_layer = std::make_unique<ushionn::nn::DenseLayer>(1, input_features, output_features,
                                                                 "dense1");  // Bias 없이 테스트 단순화

    // 레이어의 파라미터(가중치)를 미리 정의된 값으로 설정
    // get_parameters()가 Tensor&의 vector를 반환한다고 가정. 실제 Tensor 객체에 접근.
    // ---> [내가 채울 부분] get_parameters()가 반환하는 타입에 맞게 아래 코드 수정
    //      (ushionn::Tensor* 를 반환하도록 수정했다면, -> 사용)
    //      DenseLayer의 weights_ 멤버에 직접 접근하거나, set_weights 같은 메소드가 있다면 사용.
    //      여기서는 get_parameters()로 접근 가능한 ushionn::Tensor 객체의 데이터를 직접 설정한다고 가정.
    std::vector<ushionn::Tensor*> params = dense_layer->get_parameters();
    ASSERT_EQ(params.size(), 1);            // Bias가 없으므로 가중치만
    ushionn::Tensor& weights = *params[0];  // 실제 weights Tensor

    std::vector<float> W_host_data = {
        1.0f, 2.0f, 3.0f,  // 입력 1 -> 출력 1, 2, 3
        4.0f, 5.0f, 6.0f   // 입력 2 -> 출력 1, 2, 3
    };  // Shape: (input_features, output_features) = (2, 3)
        // weights Tensor는 (input_features, output_features) shape을 가져야 함.
        // weights.reshape({input_features, output_features}); // 필요시 shape 명시적 설정
    weights =
        ushionn::Tensor({input_features, output_features}, W_host_data.data(), W_host_data.size() * sizeof(float));
    weights.to_device();  // 가중치를 GPU로

    model.add_layer(std::move(dense_layer));

    // 2. 입력 데이터 준비
    int64_t batch_size = 1;
    std::vector<float> X_host_data = {10.0f, 20.0f};  // Shape: (batch_size, input_features) = (1, 2)
    ushionn::Tensor X_input({batch_size, input_features}, X_host_data.data(), X_host_data.size() * sizeof(float));
    X_input.to_device();  // 입력 데이터를 GPU로

    // 3. 모델 그래프 빌드 (입력 템플릿 사용)
    //    model_input_template은 X_input과 동일한 shape, dtype 등을 가져야 함.
    ushionn::Tensor model_input_template({batch_size, input_features}, fe::DataType_t::FLOAT, false,
                                         "graph_input_template");
    model.build_model_graph(model_input_template);

    // 4. 순전파 실행
    ushionn::Tensor Y_pred = model.predict(X_input);

    // 5. 결과 검증
    ASSERT_TRUE(Y_pred.is_on_device() ||
                Y_pred.get_data_location() == ushionn::Tensor::DataLocation::DEVICE);  // 결과는 GPU에 있어야 함
    ASSERT_EQ(Y_pred.get_shape(), std::vector<int64_t>({batch_size, output_features}));

    Y_pred.to_host();  // 결과를 CPU로 가져옴
    const float* Y_pred_host_data = static_cast<const float*>(Y_pred.get_host_ptr());
    ASSERT_NE(Y_pred_host_data, nullptr);

    // 수동 계산: Z = XW
    // X (1x2) = [10, 20]
    // W (2x3) = [[1, 2, 3],
    //            [4, 5, 6]]
    // Z (1x3) = [ (10*1 + 20*4), (10*2 + 20*5), (10*3 + 20*6) ]
    //         = [ (10 + 80),   (20 + 100),  (30 + 120)  ]
    //         = [ 90,          120,         150         ]
    std::vector<float> Y_expected_data = {90.0f, 120.0f, 150.0f};

    PrintTensorData("Input X", X_input);
    PrintTensorData("Weights W", weights);
    PrintTensorData("Predicted Y", Y_pred);

    for (size_t i = 0; i < Y_expected_data.size(); ++i)
    {
        EXPECT_TRUE(AreFloatsClose(Y_pred_host_data[i], Y_expected_data[i]))
            << "Mismatch at index " << i << ": expected " << Y_expected_data[i] << ", got " << Y_pred_host_data[i];
    }
}

// ---> [내가 채울 부분] 추가 테스트 케이스 작성:
// 1.  Dense 레이어 1개 + Bias 포함된 경우
// 2.  Dense 레이어 2개 쌓은 경우 (활성화 함수 없이)
// 3.  (구현되었다면) ReLU 같은 활성화 함수 레이어를 포함한 경우
// 4.  배치 크기가 1보다 큰 경우
// 5.  입력/출력 특징 수가 다른 다양한 경우

// TEST_F(ModelTest, SingleDenseLayerWithBias)
// {
//     // 1. 모델 구성
//     ushionn::Sequential model(cudnn_handle_);
//     int64_t input_features = 3;
//     int64_t output_features = 2;
//     // ---> DenseLayer 생성 (has_bias = true)
//     // auto dense_layer_with_bias = std::make_unique<ushionn::DenseLayer>(input_features, output_features,
//     "dense_bias",
//     // true);

//     // ---> 가중치 (W) 및 편향 (B) 데이터 설정 및 GPU로 전송
//     //      std::vector<ushionn::Tensor*> params = dense_layer_with_bias->get_parameters();
//     //      ushionn::Tensor& weights = *params[0];
//     //      ushionn::Tensor& bias = *params[1]; (순서 확인 필요)
//     //      (W_host_data, B_host_data 설정)
//     //      weights = ushionn::Tensor(...W_host_data...); weights.to_device();
//     //      bias = ushionn::Tensor(...B_host_data...); bias.to_device();

//     // model.add_layer(std::move(dense_layer_with_bias));

//     // 2. 입력 데이터 준비 (X_input) 및 GPU로 전송

//     // 3. 모델 그래프 빌드 (model_input_template)

//     // 4. 순전파 실행 (Y_pred = model.predict(X_input))

//     // 5. 결과 검증
//     //    ASSERT_EQ(Y_pred.get_shape(), std::vector<int64_t>({/*batch_size*/, output_features}));
//     //    Y_pred.to_host();
//     //    const float* Y_pred_host_data = ... ;
//     //    ---> 수동으로 Z = XW + b 계산 (Y_expected_data)
//     //    ---> Y_pred_host_data와 Y_expected_data 비교 (AreFloatsClose 사용)

//     // 이 테스트 케이스는 직접 구현해보세요!
//     GTEST_SKIP() << "Test SingleDenseLayerWithBias not yet implemented.";
// }

// TEST_F(SequentialPredictTest, TwoDenseLayers)
// {
//     // 1. 모델 구성 (Dense1 -> Dense2)
//     //    ushionn::Sequential model(cudnn_handle_);
//     //    auto dense1 = std::make_unique<ushionn::DenseLayer>(/*in1, out1*/ ... false);
//     //    auto dense2 = std::make_unique<ushionn::DenseLayer>(/*in2=out1, out2*/ ... false);
//     //    (각 레이어 가중치 설정 및 GPU 전송)
//     //    model.add_layer(std::move(dense1));
//     //    model.add_layer(std::move(dense2));

//     // 2. 입력 데이터 준비 및 GPU 전송

//     // 3. 모델 그래프 빌드

//     // 4. 순전파 실행

//     // 5. 결과 검증
//     //    Y_intermediate = X * W1
//     //    Y_final = Y_intermediate * W2
//     //    수동 계산 결과와 비교

//     GTEST_SKIP() << "Test TwoDenseLayers not yet implemented.";
// }