#include <gtest/gtest.h>

#include "core/common.h"
#include "cuda/cuda_utils.h"
#include "model/model.h"

namespace fe = cudnn_frontend;

class ModelTest : public ::testing::Test
{
   protected:
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

    void SetUp() override { ushionn::cuda::utils::printGpuMemoryUsage("Before predict test case"); }

    void TearDown() override { ushionn::cuda::utils::printGpuMemoryUsage("After predict test case"); }
};

TEST_F(ModelTest, SingleDenseLayerNoActivation)
{
    // 1. 모델 구성
    ushionn::model::Sequential model;  // 모델에 cudnnHandle 전달
    // ---> [내가 채울 부분] 아래 DenseLayer 생성자 인자를 실제 구현에 맞게 수정.
    //      (input_size, output_size, name, has_bias=false 등)
    //      예: auto dense_layer = std::make_unique<ushionn::DenseLayer>(2, 3, "dense1", false); // 입력 2, 출력 3, bias
    //      없음
    int64_t batch_size = 1;
    int64_t input_features = 2;
    int64_t output_features = 3;
    auto dense_layer = std::make_unique<ushionn::nn::DenseLayer>(batch_size, input_features, output_features, "dense1");

    // 레이어의 파라미터(가중치)를 미리 정의된 값으로 설정
    std::vector<ushionn::Tensor*> params = dense_layer->get_parameters();
    ushionn::Tensor& weights = *params[0];  // 실제 weights Tensor
    ushionn::Tensor& bias = *params[1];     // bias Tensor

    std::vector<float> W_host_data = {1.0f, 2.0f, 3.0f,
                                      4.0f, 5.0f, 6.0f};  // Shape: (input_features, output_features) = (2, 3)
                                                          // weights Tensor는 (input_features, output_features) shape을
                                                          // 가져야 함. weights.reshape({input_features,
                                                          // output_features});
    std::vector<float> b_host_data = {1.f, 1.f, 1.f};
    weights = ushionn::Tensor({batch_size, input_features, output_features}, W_host_data.data(),
                              W_host_data.size() * sizeof(float));
    weights.to_device();  // 가중치를 GPU로

    bias = ushionn::Tensor({batch_size, 1, output_features}, b_host_data.data(), b_host_data.size() * sizeof(float));
    bias.to_device();

    model.add_layer(std::move(dense_layer));

    // 2. 입력 데이터 준비

    std::vector<float> X_host_data = {10.0f, 20.0f};  // Shape: (batch_size, input_features) = (1, 2)
    ushionn::Tensor X_input({batch_size, 1, input_features}, X_host_data.data(), X_host_data.size() * sizeof(float));
    X_input.to_device();  // 입력 데이터를 GPU로

    // 3. 모델 그래프 빌드 (입력 템플릿 사용)
    //    model_input_template은 X_input과 동일한 shape, dtype 등을 가져야 함.
    ushionn::Tensor model_input_template({batch_size, 1, input_features}, fe::DataType_t::FLOAT, false,
                                         "graph_input_template");
    ASSERT_TRUE(model.build_model_graph(model_input_template));

    // 4. 순전파 실행
    ushionn::Tensor Y_pred = model.predict(X_input);

    std::cout << "predict complete" << std::endl;

    // 5. 결과 검증
    ASSERT_TRUE(Y_pred.is_on_device() ||
                Y_pred.get_data_location() == ushionn::Tensor::DataLocation::DEVICE);  // 결과는 GPU에 있어야 함
    ASSERT_EQ(Y_pred.get_shape(), std::vector<int64_t>({batch_size, 1, output_features}));

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
    std::vector<float> Y_expected_data = {91.0f, 121.0f, 151.0f};

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