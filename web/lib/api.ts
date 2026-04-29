import axios from "axios";

let accessToken: string | null = null; // Token lưu trong bộ nhớ (mất khi reload)
let isRefreshing = false; // Cờ quá trình lấy token mới
let failedQueue: any[] = []; // Hàng đợi các request bị tạm dừng để chờ token mới
let onAuthFailure: (() => void) | null = null; // Callback khi auth thất bại

// Khi có token mới thì chạy tiếp (resolve), nếu lỗi thì hủy (reject)
const processQueue = (error: any, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) prom.reject(error);
    else prom.resolve(token);
  });
  failedQueue = [];
};

export const setOnAuthFailure = (callback: (() => void) | null) => {
  onAuthFailure = callback;
};

export const api = axios.create({
  baseURL: "/api",
  withCredentials: true,
  headers: { "Content-Type": "application/json" },
});

// Tự động gắn Access Token vào mỗi request
api.interceptors.request.use((config) => {
  if (accessToken) {
    config.headers.Authorization = `Bearer ${accessToken}`;
  }
  return config;
});

// Xử lý khi Access Token hết hạn (Lỗi 401)
api.interceptors.response.use(
  (response) => response, // Nếu thành công thì trả về
  async (error) => {
    const originalRequest = error.config;

    // Nếu lỗi 401 (Unauthorized) và chưa từng thử refresh trước đó
    if (
      (error.response?.status === 401 || error.response?.status === 403) &&
      !originalRequest._retry
    ) {
      // Nếu đã có một request khác đang đi refresh => hàng đợi
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then((token) => {
            originalRequest.headers.Authorization = `Bearer ${token}`;
            return api(originalRequest);
          })
          .catch((err) => Promise.reject(err));
      }

      // Request đầu tiên phát hiện token hết hạn
      originalRequest._retry = true;
      isRefreshing = true;

      try {
        // Gọi API Refresh lấy Access Token mới
        const res = await axios.post(
          "/api/auth/refresh",
          {},
          { withCredentials: true },
        );
        const newAccessToken = res.data.access_token;
        setAccessToken(newAccessToken);
        processQueue(null, newAccessToken); /// Thông báo cho hàng đợi đã có token

        originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
        return api(originalRequest); // Thực thi lại chính request bị lỗi ban đầu
      } catch (refreshError) {
        processQueue(refreshError, null);
        setAccessToken(null);
        // Notify AuthContext that authentication has failed
        if (onAuthFailure) {
          onAuthFailure();
        }
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }
    return Promise.reject(error);
  },
);

export const setAccessToken = (token: string | null) => {
  accessToken = token;
};
