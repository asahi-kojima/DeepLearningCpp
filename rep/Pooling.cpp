#include "Pooling.hpp"

//******************************
//       初期化関数
//******************************
void Pooling::initialize()
{
    ;
}

void Pooling::forward(void * void_input)
{
    double ** input = (double**)void_input;
#ifdef _OPENMP
    #pragma omp pallel for
#endif
    for (int N = 0; N < m_batchNum; N++)
    {
        m_inputData[N] = input[N];
        for (int i = 0; i < (m_outputChannel * m_outputHeight * m_outputWidth); i++)
        {
            int c = i / (m_outputHeight * m_outputWidth);
            int h = (i - c * (m_outputHeight * m_outputWidth)) / m_outputWidth;
            int w = i % m_outputWidth;

            m_resultMemo[N][i] = 0;

            int _H = (h * m_stride + 0 - m_pad);
            int _W = (w * m_stride + 0 - m_pad);
            
            double max = 0;
            if (_H < 0 || _H >= m_inputHeight || _W < 0 || _W >= m_inputWidth)
            {
                max = 0;
            }
            else 
            {
                max = m_inputData[N][c * (m_inputHeight * m_inputWidth) + (_H + 0) * (m_inputWidth) + (_W + 0)];
            }
            
            for (int H = 0; H < m_filterHeight; H++)
            {
                for (int W = 0; W < m_filterWidth; W++)
                {
                    double tmp = 0;
                    if (_H + H < 0 || _H + H >= m_inputHeight || _W + W < 0 || _W + W >= m_inputWidth)
                    {
                        tmp = 0;
                    }
                    else
                    {
                        tmp = m_inputData[N][c * (m_inputHeight * m_inputWidth) + (_H + H) * (m_inputWidth) + (_W + W)];
                    }

                    if (tmp > max)
                    {
                        max = tmp;
                        m_resultMemo[N][i] = H * m_filterWidth + W;
                    }
                }
            }
            m_forwardResult[N][i] = max;
        }
        input[N] = m_forwardResult[N];
    }
}

void Pooling::backward(void * void_dout)
{
    double ** dout = (double **)void_dout;

#ifdef _OPENMP
    #pragma omp pallel for
#endif
    for (int N = 0; N < m_batchNum; N++)
    {
        for (int i = 0; i < (m_inputChannel * m_inputHeight * m_inputWidth); i++)
        {
            m_backwardResult[N][i] = 0;
        }
        for (int i = 0; i < (m_outputChannel * m_outputHeight * m_outputWidth); i++)
        {
            //m_backwardResult[N][i] = 0;
        
            int c = i / (m_outputHeight * m_outputWidth);
            int h = (i - c * (m_outputHeight * m_outputWidth)) / m_outputWidth;
            int w = i % m_outputWidth;

            int tmp = m_resultMemo[N][i];
            int H = tmp / m_filterWidth;
            int W = tmp % m_filterWidth;
            
            int _H = (h * m_stride + 0 - m_pad);
            int _W = (w * m_stride + 0 - m_pad);

            if (!(_H + H < 0 || _H + H >= m_inputHeight || _W + W < 0 || _W + W >= m_inputWidth))
            {
                m_backwardResult[N][c * (m_inputHeight * m_inputWidth) + (_H + H) * (m_inputWidth) + (_W + W)] = dout[N][i];
            }
        }

        dout[N] = m_backwardResult[N];
    }
}