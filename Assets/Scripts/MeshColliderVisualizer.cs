using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class MeshColliderVisualizer : MonoBehaviour
{
    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;

        Mesh mesh = GetComponent<MeshFilter>().sharedMesh;

        if (mesh != null)
        {
            Vector3[] vertices = mesh.vertices;
            int[] triangles = mesh.triangles;

            for (int i = 0; i < triangles.Length; i += 3)
            {
                Vector3 v1 = transform.TransformPoint(vertices[triangles[i]]);
                Vector3 v2 = transform.TransformPoint(vertices[triangles[i + 1]]);
                Vector3 v3 = transform.TransformPoint(vertices[triangles[i + 2]]);

                Gizmos.DrawLine(v1, v2);
                Gizmos.DrawLine(v2, v3);
                Gizmos.DrawLine(v3, v1);
            }
        }
    }
}
