import axios from "axios";

let accessToken: string | null = null;

export const api = axios.create({
  baseURL: "/api",
  withCredentials: true,
  headers: { "Content-Type": "application/json" },
});

// Tự động gắn Access Token vào mỗi request
api.interceptors.request.use(
  (config) => {
    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Xử lý khi Access Token hết hạn (Lỗi 401)
api.interceptors.response.use(
  (response) => response, // Nếu thành công thì trả về luôn
  async (error) => {
    const originalRequest = error.config;

    // Nếu lỗi 401 (Unauthorized) và chưa từng thử refresh trước đó
    if ((error.response?.status === 401 || error.response?.status === 403) && !originalRequest._retry) {
      originalRequest._retry = true;
      try {
        // Gọi endpoint refresh để lấy access_token mới
        const res = await axios.post("/api/auth/refresh", {}, { withCredentials: true });
        const newAccessToken = res.data.access_token;
        accessToken = newAccessToken; // Cập nhật token mới vào bộ nhớ

        // Thay đổi header của request cũ và thực hiện lại
        originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
        return api(originalRequest);
      } catch (refreshError) {
        accessToken = null;
        return Promise.reject(refreshError);
      }
    }
    return Promise.reject(error);
  }
);

export const setAccessToken = (token: string | null) => {
  accessToken = token;
};