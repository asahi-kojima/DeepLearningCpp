#include "Deconvolution.hpp"
#include <time.h>

//******************************
//       初期化関数
//******************************
void Deconvolution::initialize()
{
    ;
}

void isRangeCorrect(int n, int min, int max, int ID)
{
    if (!(n >= min && n < max))
    {
        std::cout << "Range Error" << ID << std::endl;
    }
}

//******************************
//       順伝搬関数　実装
//******************************
void Deconvolution::forward(void * void_input)
{
    double ** input = (double **) void_input;

#ifdef _OPENMP
    #pragma omp pallel for;
#endif
    for (int N = 0; N < m_batchNum; N++)
    {
        int Ph = m_Fh - 1 - m_pad;
        int Pw = m_Fw - 1 - m_pad;

        for (int _IcIhIw = 0; _IcIhIw < m_IcIhIw; _IcIhIw++)
        {
            int _Ic = _IcIhIw / m_IhIw;
            int _Ih = (_IcIhIw - _Ic * m_IhIw) / m_Iw;
            int _Iw = _IcIhIw % m_Ih;

            int startPoint_FH = Ph + _Ih * m_stride + 1 - m_Fh;
            int endPoint_FH = startPoint_FH + m_Fh;
            int startPoint_FW = Pw + _Iw * m_stride + 1 - m_Fw;
            int endPoint_FW = startPoint_FW + m_Fw;

            int trueStartPoint_FH = ((0 > startPoint_FH) ? 0 : startPoint_FH);
            int trueEndPoint_FH = ((m_Oh <= endPoint_FH) ? m_Oh : endPoint_FH);
            int trueStartPoint_FW = ((0 > startPoint_FW) ? 0 : startPoint_FW); 
            int trueEndPoint_FW = ((m_Ow <= endPoint_FW) ? m_Ow : endPoint_FW);
            
            int detailPos_H = (m_Fh - 1) - (trueStartPoint_FH - startPoint_FH);
            int detailPos_W = (m_Fw - 1) - (trueStartPoint_FW - startPoint_FW);

            for (int _Fh = trueStartPoint_FH; _Fh < trueEndPoint_FH; _Fh++)
            {
                for (int _Fw = trueStartPoint_FW; _Fw < trueEndPoint_FW; _Fw++)
                {
                    int index_col = (_Fh * m_Ow + _Fw);
                    int offset_H = _Fh - trueStartPoint_FH;
                    int offset_W = _Fw - trueStartPoint_FW;
                    int index_row = (_Ic) * (m_FhFw) + (detailPos_H - offset_H) * m_Fw + (detailPos_W - offset_W);
                    int index = index_col * (m_FcFhFw) + index_row;
#ifdef ___RANGE_CHECK
                    isRangeCorrect(index, 0, m_OhOw * m_FcFhFw, 0);
                    isRangeCorrect(_IcIhIw, 0, m_IcIhIw, 1);
                    isRangeCorrect(_IcIhIw, 0, m_IcIhIw, 2);
                    isRangeCorrect(m_projectionIndex[N][_IcIhIw], 0, m_OhOw, 3);
                    isRangeCorrect(_IcIhIw, 0, m_IcIhIw, 4);
#endif
                    m_reshapedInputData[N][index] = input[N][_IcIhIw];
                    if (!flag)
                    {
                        m_projectionList[N][_IcIhIw][m_projectionIndex[N][_IcIhIw]] = index;
                        m_projectionIndex[N][_IcIhIw]++;
                    }
                }
            }
        }

        for (int _OcOhOw = 0; _OcOhOw < m_OcOhOw; _OcOhOw++)
        {
            int _Oc = _OcOhOw / m_OhOw;
            int _OhOw = _OcOhOw - _Oc * m_OhOw;

            double tmp = 0;
            for (int k = 0; k < m_FcFhFw; k++)
            {
#ifdef ___RANGE_CHECK
                    isRangeCorrect(_Oc * m_FcFhFw + k, 0, m_Fn * m_Fc * m_Fh * m_Fw, 5);
                    isRangeCorrect(_OhOw * m_FcFhFw + k, 0, m_OhOw * m_FcFhFw, 6);
#endif
                tmp += m_matrix[_Oc * m_FcFhFw + k] * m_reshapedInputData[N][_OhOw * m_FcFhFw + k];
            }
#ifdef ___RANGE_CHECK
                    isRangeCorrect(_OcOhOw, 0, m_OcOhOw, 7);
                    isRangeCorrect(_Oc, 0, m_Oc, 8);
#endif            
            m_forwardResult[N][_OcOhOw] = tmp + m_bias[_Oc];
        }

        input[N] = m_forwardResult[N];
    }
    flag = true;
}


//******************************
//       逆伝搬関数　実装
//******************************
void Deconvolution::backward(void * void_dout)
{
    double ** dout = (double **)void_dout;

#ifdef _OPENMP
    #pragma omp pallel for;
#endif
    for (int _Oc = 0; _Oc < m_Oc; _Oc++)
    {
        double tmp = 0;
        for (int N = 0; N < m_batchNum; N++)
        {
            for (int _OhOw = 0; _OhOw < m_OhOw; _OhOw++)
            {
#ifdef ___RANGE_CHECK
                    isRangeCorrect(_Oc * m_OhOw + _OhOw, 0, m_OcOhOw, 0);
#endif
                tmp += dout[N][_Oc * m_OhOw + _OhOw];
            }
        }
#ifdef ___RANGE_CHECK
                    isRangeCorrect(_Oc, 0, m_Oc, 1);
#endif        
        m_dBias[_Oc] = tmp;
    }


#ifdef _OPENMP
    #pragma omp pallel for;
#endif
    for (int index = 0; index < m_Fn * m_FcFhFw; index++)
    {
        double tmp = 0;
        int x = index / m_FcFhFw;
        int y = index % m_FcFhFw;
        for (int N = 0; N < m_batchNum; N++)
        {
            for (int _OhOw = 0; _OhOw < m_OhOw; _OhOw++)
            {
#ifdef ___RANGE_CHECK
                    isRangeCorrect(x * m_OhOw + _OhOw, 0, m_OcOhOw, 2);
                    isRangeCorrect(_OhOw * m_FcFhFw + y, 0, m_OhOw * m_FcFhFw, 3);
#endif  
                tmp += dout[N][x * m_OhOw + _OhOw] * m_reshapedInputData[N][_OhOw * m_FcFhFw + y];
            } 
        }
#ifdef ___RANGE_CHECK
                    isRangeCorrect(index, 0, m_Fn * m_FcFhFw, 4);
#endif  
        m_dMatrix[index] = tmp;
    }


#ifdef _OPENMP
    #pragma omp pallel for;
#endif
    for (int N = 0; N < m_batchNum; N++)
    {
        for (int _IcIhIw = 0; _IcIhIw < m_IcIhIw; _IcIhIw++)
        {
            double tmp = 0;
            for (int index = 0; index < m_projectionIndex[N][_IcIhIw]; index++)
            {
#ifdef ___RANGE_CHECK
                isRangeCorrect(_IcIhIw, 0, m_IcIhIw, 5);
                isRangeCorrect(index, 0, m_FhFw, 6);
#endif  
                int Index = m_projectionList[N][_IcIhIw][index];
                int Index_H = Index / m_FcFhFw;
                int Index_W = Index % m_FcFhFw;
                for (int _Fn = 0; _Fn < m_Fn; _Fn++)
                {
#ifdef ___RANGE_CHECK
                    isRangeCorrect(_Fn * m_OhOw + Index_H, 0, m_OcOhOw, 7);
                    isRangeCorrect(_Fn * m_FcFhFw + Index_W, 0, m_Fn * m_FcFhFw, 8);
#endif  
                    tmp += dout[N][_Fn * m_OhOw + Index_H] * m_matrix[_Fn * m_FcFhFw + Index_W];
                }
            }
            m_backwardResult[N][_IcIhIw] = tmp;
        }
        dout[N] = m_backwardResult[N];
    }
}


